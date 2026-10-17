#include "nvof/gpu_engine.hpp"
#include "nvof/gpu_wait_service.hpp"
#include "nvof/motion_synthesizer.hpp"
#include <d3d11_4.h>
#include "nvof/scene_cut.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <d3d10_1.h>
#include "p010_shaders.hpp"
#include "scene_shaders.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace nvof {
namespace {
using Microsoft::WRL::ComPtr;
void hr_check(HRESULT hr,const char* action) { if(FAILED(hr)) throw std::runtime_error(std::string(action)+" failed, HRESULT="+std::to_string(static_cast<unsigned long>(hr))); }
// A diagnostic allocation failure must never reset or stall interpolation.
std::shared_ptr<const RifeTiming> snapshot_rife_timing(const RifeTiming& timing) noexcept {
    try { return std::make_shared<RifeTiming>(timing); } catch (...) { return {}; }
}
DXGI_FORMAT texture_format(const GpuFrame& frame) {D3D11_TEXTURE2D_DESC desc{};frame.texture->GetDesc(&desc);return desc.Format;}
DXGI_FORMAT plane_format(DXGI_FORMAT format,unsigned plane) {
    return format==DXGI_FORMAT_P010?(plane?DXGI_FORMAT_R16G16_UNORM:DXGI_FORMAT_R16_UNORM):(plane?DXGI_FORMAT_R8G8_UNORM:DXGI_FORMAT_R8_UNORM);
}

std::atomic<unsigned> retired_engines{0};
struct RetiredEngine {
    std::shared_ptr<void> host_lifetime;
    std::unique_ptr<GpuRifeEngine> engine;
    HMODULE module=nullptr;
};
DWORD WINAPI retire_engine_entry(void* value){
    HMODULE module=nullptr;
    {
        std::unique_ptr<RetiredEngine> retired(static_cast<RetiredEngine*>(value));
        module=retired->module;
        // Avoid holding the renderer's D3D lock while the loader serializes
        // CUDA destruction calls. The engine/stream/buffers stay owned here.
        while(MotionSynthesizer::optional_preload_pending())Sleep(2);
        retired->engine.reset();
    }
    --retired_engines;
    FreeLibraryAndExitThread(module,0);
}


}

struct GpuRifeEngine::Impl {
    std::mutex mutex;
    bool trace_timing=false;
    bool trace_rife=false;
    unsigned profile_interval=1;uint64_t profile_sequence=0;
    GpuPairTiming timing{};
    GpuCompletionMode completion_mode=GpuCompletionMode::blocking;
    HANDLE decoder_mutex=nullptr; // Borrowed, never closed here.
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11DeviceContext1> context1;
    ComPtr<ID3D10Multithread> multithread;
    ComPtr<ID3DDeviceContextState> private_state;
    ComPtr<ID3D11Query> completion;
    ComPtr<ID3D11DeviceContext3> context3;
    HANDLE flush_event=nullptr;
    ComPtr<ID3D11DeviceContext4> context4;
    ComPtr<ID3D11Fence> completion_fence;
    HANDLE completion_event=nullptr;uint64_t fence_value=0;
    GpuRifeEngine::WaitService wait_service;
    ComPtr<ID3D11ComputeShader> scene_compute,scene_identity,scene_match;
    SceneDetector scene_detector;
    ID3D11Texture2D* scene_previous=nullptr; // Identity only, no extra decoder lease.
    int64_t scene_previous_pts=-1;UINT scene_previous_slice=0;
    ComPtr<ID3D11Buffer> scene_buffer,scene_readback;
    ComPtr<ID3D11UnorderedAccessView> scene_uav;
    struct OutputSlot {DXGI_FORMAT format=DXGI_FORMAT_NV12;ComPtr<ID3D11Texture2D> texture;std::array<ComPtr<ID3D11ShaderResourceView>,2> views;std::array<ComPtr<ID3D11RenderTargetView>,2> targets;int width=0,height=0;};
    std::vector<std::shared_ptr<OutputSlot>> output_pool;
    ComPtr<ID3D11VertexShader> vertex_shader;
    ComPtr<ID3D11PixelShader> p010_pixel_shader;
    ComPtr<ID3D11Texture2D> p010_capture;
    std::array<ComPtr<ID3D11ShaderResourceView>,2> p010_views;
    int p010_width=0,p010_height=0;
    ComPtr<ID3D11Buffer> scene_constants;
    ComPtr<ID3D11RasterizerState> rasterizer;
    std::string name;
    std::unique_ptr<MotionSynthesizer> native_synthesizer;
    int64_t cached_pts=0;
    ComPtr<ID3D11Texture2D> cached_texture;
    std::shared_ptr<void> cached_lease;
    UINT cached_slice=0;

    struct DeviceScope {
        Impl& owner;bool owns_mutex=false,owns_multithread=false;ComPtr<ID3DDeviceContextState> old_state;
        bool released=true;
        explicit DeviceScope(Impl& o):owner(o){acquire();}
        void acquire(){
            if(!released)return;
            using Clock=std::chrono::steady_clock;
            auto tick=owner.trace_timing?Clock::now():Clock::time_point{};
            auto mark=[&](double& elapsed){if(owner.trace_timing){const auto now=Clock::now();elapsed+=std::chrono::duration<double,std::milli>(now-tick).count();tick=now;}};
            if(owner.decoder_mutex){const DWORD wait=WaitForSingleObject(owner.decoder_mutex,10000);if(wait!=WAIT_OBJECT_0&&wait!=WAIT_ABANDONED)throw std::runtime_error("Timed out acquiring decoder D3D11 mutex");owns_mutex=true;}
            mark(owner.timing.decoder_lock_ms);
            // MF decoders know the device manager, not our renderer mutex.
            // Hold the protected D3D context for the complete save/work/restore
            // sequence; per-call protection alone permits state interleaving.
            if(owner.multithread && owner.multithread->GetMultithreadProtected()) {
                owner.multithread->Enter();owns_multithread=true;
            }
            mark(owner.timing.multithread_lock_ms);
            owner.context1->SwapDeviceContextState(owner.private_state.Get(),&old_state);
            mark(owner.timing.state_swap_ms);
            released=false;
        }
        void release() noexcept {
            if(released)return;
            owner.context1->SwapDeviceContextState(old_state.Get(),nullptr);
            old_state.Reset();
            if(owns_multithread){owner.multithread->Leave();owns_multithread=false;}
            if(owns_mutex){ReleaseMutex(owner.decoder_mutex);owns_mutex=false;}
            released=true;
        }
        ~DeviceScope(){release();}
    };

    Impl(const std::filesystem::path& directory,ID3D11Device* d,ID3D11DeviceContext* c,HANDLE shared_mutex,GpuCompletionMode mode):completion_mode(mode),decoder_mutex(shared_mutex),device(d),context(c){
        try{
            if(!d||!c)throw std::invalid_argument("A decoder D3D11 device and immediate context are required");
            if(c->GetType()!=D3D11_DEVICE_CONTEXT_IMMEDIATE)throw std::invalid_argument("D3D11 immediate context required");
            ComPtr<ID3D11Device> context_device;c->GetDevice(&context_device);if(context_device.Get()!=d)throw std::invalid_argument("D3D11 context belongs to another device");
            hr_check(context.As(&context1),"Query ID3D11DeviceContext1");
            context.As(&multithread); // Optional for existing non-MF callers.
            ComPtr<ID3D11Device1> device1;hr_check(device.As(&device1),"Query ID3D11Device1");
            const D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};D3D_FEATURE_LEVEL actual{};
            hr_check(device1->CreateDeviceContextState(0,levels,2,D3D11_SDK_VERSION,__uuidof(ID3D11Device),&actual,&private_state),"CreateDeviceContextState");
            D3D11_QUERY_DESC query_desc{D3D11_QUERY_EVENT,0};hr_check(device->CreateQuery(&query_desc,&completion),"Create GPU completion query");
            // A context event marks completion of the submitted D3D11 work
            // without inserting a cross-API fence signal into the host queue.
            // Keep ALL contexts: decode/copy/compute dependencies must not be
            // dropped merely to make the first analysis wait appear shorter.
            if(completion_mode==GpuCompletionMode::context_ordered&&SUCCEEDED(context.As(&context3)))
                flush_event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
            ComPtr<ID3D11Device5> device5;
            if(SUCCEEDED(device.As(&device5))&&SUCCEEDED(context.As(&context4))&&SUCCEEDED(device5->CreateFence(0,D3D11_FENCE_FLAG_NONE,IID_PPV_ARGS(&completion_fence)))){
                completion_event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!completion_event)completion_fence.Reset();
            }
            hr_check(device->CreateComputeShader(shaders::scene::gridProbe,sizeof(shaders::scene::gridProbe),nullptr,&scene_compute),"Create scene detector");
            hr_check(device->CreateComputeShader(shaders::scene::identity,sizeof(shaders::scene::identity),nullptr,&scene_identity),"Create exact identity fallback");
            hr_check(device->CreateComputeShader(shaders::scene::match,sizeof(shaders::scene::match),nullptr,&scene_match),"Create motion tolerant scene detector");
            D3D11_BUFFER_DESC stats_desc{};stats_desc.ByteWidth=sizeof(SceneStats)+sizeof(uint32_t)+scene_grid_count*sizeof(uint32_t);stats_desc.Usage=D3D11_USAGE_DEFAULT;stats_desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;stats_desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;stats_desc.StructureByteStride=4;
            hr_check(device->CreateBuffer(&stats_desc,nullptr,&scene_buffer),"Create scene statistics");
            D3D11_UNORDERED_ACCESS_VIEW_DESC stats_view{};stats_view.Format=DXGI_FORMAT_UNKNOWN;stats_view.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;stats_view.Buffer.NumElements=stats_desc.ByteWidth/sizeof(uint32_t);
            hr_check(device->CreateUnorderedAccessView(scene_buffer.Get(),&stats_view,&scene_uav),"Create scene statistics view");
            stats_desc.ByteWidth=sizeof(SceneStats)+sizeof(uint32_t);
            stats_desc.Usage=D3D11_USAGE_STAGING;stats_desc.BindFlags=0;stats_desc.MiscFlags=0;stats_desc.StructureByteStride=0;stats_desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            hr_check(device->CreateBuffer(&stats_desc,nullptr,&scene_readback),"Create scene statistics readback");
            hr_check(device->CreateVertexShader(shaders::p010::vs,sizeof(shaders::p010::vs),nullptr,&vertex_shader),"CreateVertexShader");
            D3D11_BUFFER_DESC cb{};cb.ByteWidth=16;cb.Usage=D3D11_USAGE_DEFAULT;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            hr_check(device->CreateBuffer(&cb,nullptr,&scene_constants),"Create scene dimensions");
            D3D11_RASTERIZER_DESC raster{};raster.FillMode=D3D11_FILL_SOLID;raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=TRUE;
            hr_check(device->CreateRasterizerState(&raster,&rasterizer),"CreateRasterizerState");
            {
                DeviceScope scope(*this);
                native_synthesizer=std::make_unique<MotionSynthesizer>(device.Get(),context.Get(),directory);
                ComPtr<IDXGIDevice> dxgi;hr_check(device.As(&dxgi),"Native DXGI device");
                ComPtr<IDXGIAdapter> adapter;hr_check(dxgi->GetAdapter(&adapter),"Native adapter");
                DXGI_ADAPTER_DESC desc{};hr_check(adapter->GetDesc(&desc),"Native adapter name");
                const int bytes=WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,nullptr,0,nullptr,nullptr);
                if(bytes<=0)throw std::runtime_error("Native adapter name conversion failed");
                name.resize(bytes);WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name.data(),bytes,nullptr,nullptr);
                name.pop_back();
                return;
            }
        }catch(...){cleanup();throw;}
    }
    ~Impl(){cleanup();}
    void cleanup() noexcept {
        if(native_synthesizer) {
            try {DeviceScope scope(*this);wait_gpu();native_synthesizer.reset();output_pool.clear();}
            catch(...) {
                native_synthesizer.release(); // Host lock is unavailable: do not race its context.
                OutputDebugStringA("RIFE: shared D3D11 lock unavailable during native teardown\n");
                return;
            }
        }
        if(completion_event){CloseHandle(completion_event);completion_event=nullptr;}
        if(flush_event){CloseHandle(flush_event);flush_event=nullptr;}
    }
    void validate(const GpuFrame& f,bool allow_p010=true) const {
        if(!f.texture||f.width<2||f.height<2||f.width>8192||f.height>8192||(f.width&1)||(f.height&1))throw std::invalid_argument("GPU frame must be even-size NV12");
        D3D11_TEXTURE2D_DESC desc{};f.texture->GetDesc(&desc);
        if((desc.Format!=DXGI_FORMAT_NV12&&!(allow_p010&&desc.Format==DXGI_FORMAT_P010))||desc.Width<UINT(f.width)||desc.Height<UINT(f.height)||desc.ArraySize<=f.array_slice||desc.SampleDesc.Count!=1||desc.MipLevels!=1)throw std::invalid_argument("GPU NV12 texture format, dimensions or array slice invalid");
        ComPtr<ID3D11Device> source_device;f.texture->GetDevice(&source_device);if(source_device.Get()!=device.Get())throw std::invalid_argument("GPU input texture belongs to another D3D11 device");
    }
    // A terminal completion wait needs the shared context only to submit its
    // fence/query. Restore the host's state before releasing the device locks,
    // then keep waiting for exactly the same GPU completion boundary. The
    // engine mutex and output leases remain held; neither samples nor mutable
    // engine state can be reused while the renderer prepares unrelated work.
    void wait_gpu(DeviceScope* terminal_scope=nullptr){
        using Clock=std::chrono::steady_clock;
        const bool measure=trace_timing&&terminal_scope;
        const auto submitted=measure?Clock::now():Clock::time_point{};
        if(context3&&flush_event){
            if(measure){++timing.event_wait_calls;++timing.flush_event_wait_calls;}
            // Single in-flight auto-reset event, serialized by the engine
            // mutex. The wait consumes it before any subsequent Flush1.
            context3->Flush1(D3D11_CONTEXT_TYPE_ALL,flush_event);
            if(terminal_scope)terminal_scope->release();
            const auto released=measure?Clock::now():Clock::time_point{};
            if(measure)timing.wait_submit_ms+=std::chrono::duration<double,std::milli>(released-submitted).count();
            const DWORD result=wait_for_completion(flush_event,terminal_scope);
            if(result!=WAIT_OBJECT_0){
                // Never reuse an event whose old completion may arrive later.
                // Keep the handle alive until teardown and use the existing
                // fence/query path if the caller attempts recovery.
                context3.Reset();
                throw std::runtime_error("D3D11 context completion event timeout");
            }
            hr_check(device->GetDeviceRemovedReason(),"D3D11 context completion device state");
            if(measure){
                const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-released).count();
                timing.event_sleep_ms+=elapsed;timing.wait_completion_ms+=elapsed;
            }
            return;
        }
        if(completion_fence&&completion_event){
            if(measure)++timing.event_wait_calls;
            const auto value=++fence_value;hr_check(context4->Signal(completion_fence.Get(),value),"Signal D3D11 completion fence");context->Flush();
            if(terminal_scope)terminal_scope->release();
            const auto released=measure?Clock::now():Clock::time_point{};
            if(measure)timing.wait_submit_ms+=std::chrono::duration<double,std::milli>(released-submitted).count();
            if(completion_fence->GetCompletedValue()<value){
                const auto arm=measure?Clock::now():Clock::time_point{};
                hr_check(completion_fence->SetEventOnCompletion(value,completion_event),"Set D3D11 completion event");
                const auto sleeping=measure?Clock::now():Clock::time_point{};
                if(wait_for_completion(completion_event,terminal_scope)!=WAIT_OBJECT_0)throw std::runtime_error("D3D11 completion fence timeout");
                if(measure){
                    timing.event_arm_ms+=std::chrono::duration<double,std::milli>(sleeping-arm).count();
                    timing.event_sleep_ms+=std::chrono::duration<double,std::milli>(Clock::now()-sleeping).count();
                }
            }
            hr_check(device->GetDeviceRemovedReason(),"D3D11 completion device state");
            if(measure)timing.wait_completion_ms+=std::chrono::duration<double,std::milli>(Clock::now()-released).count();
            return;
        }
        if(measure)++timing.query_wait_calls;
        context->End(completion.Get());context->Flush();
        if(terminal_scope)terminal_scope->release();
        const auto released=measure?Clock::now():Clock::time_point{};
        if(measure)timing.wait_submit_ms+=std::chrono::duration<double,std::milli>(released-submitted).count();
        const auto begin=GetTickCount64();BOOL ready=FALSE;
        for(;;){
            HRESULT hr;
            if(terminal_scope){DeviceScope poll(*this);hr=context->GetData(completion.Get(),&ready,sizeof(ready),D3D11_ASYNC_GETDATA_DONOTFLUSH);}
            else hr=context->GetData(completion.Get(),&ready,sizeof(ready),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            hr_check(hr,"D3D11 GPU completion");if(hr==S_OK&&ready){
                if(measure)timing.wait_completion_ms+=std::chrono::duration<double,std::milli>(Clock::now()-released).count();
                return;
            }
            if(GetTickCount64()-begin>10000)throw std::runtime_error("D3D11 GPU completion timeout");
            if(terminal_scope&&wait_service)service_outputs();
            SwitchToThread();
        }
    }
    HANDLE service_outputs() {
        const auto begin=trace_timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        HANDLE event=wait_service();
        if(trace_timing){++timing.wait_service_calls;timing.wait_service_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
        return event;
    }
    DWORD wait_for_completion(HANDLE event,DeviceScope* released_scope) {
        if(!released_scope||!wait_service)return WaitForSingleObject(event,10000);
        return wait_gpu_event(event,[this]{return service_outputs();},10000);
    }
    void wait_cuda_ready(DeviceScope& scope,HANDLE event){
        using Clock=std::chrono::steady_clock;
        const auto begin=trace_timing?Clock::now():Clock::time_point{};
        if(trace_timing){++timing.shared_wait_calls;++timing.event_wait_calls;++timing.cuda_ready_wait_calls;}
        scope.release();
        DWORD result=WAIT_FAILED;
        try{result=wait_for_completion(event,&scope);}catch(...){scope.acquire();throw;}
        const auto resume=trace_timing?Clock::now():Clock::time_point{};
        scope.acquire();
        if(trace_timing){
            const auto end=Clock::now();
            const double waiting=std::chrono::duration<double,std::milli>(resume-begin).count();
            const double acquiring=std::chrono::duration<double,std::milli>(end-resume).count();
            timing.wait_completion_ms+=waiting;timing.context_resume_ms+=acquiring;timing.shared_wait_ms+=waiting+acquiring;
            timing.cuda_ready_wait_ms+=waiting;timing.cuda_ready_resume_ms+=acquiring;
            if(timing.shared_wait_calls<=timing.wait_each_ms.size()){timing.wait_each_ms[timing.shared_wait_calls-1]=waiting;timing.resume_each_ms[timing.shared_wait_calls-1]=acquiring;}
        }
        if(result!=WAIT_OBJECT_0)throw std::runtime_error("RIFE CUDA readiness timeout");
    }
    void wait_and_resume(DeviceScope& scope){
        const double previous_wait=timing.wait_completion_ms;
        const auto begin=trace_timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        if(trace_timing)++timing.shared_wait_calls;
        // Only the device/context locks are released. The engine mutex keeps
        // analysis resources, RIFE execution context, sources and output leases private.
        // Resume captures the renderer's NEW state, not the pre-wait snapshot.
        try {wait_gpu(&scope);}catch(...){scope.acquire();throw;}
        const auto resume=trace_timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        scope.acquire();
        if(trace_timing){
            const auto end=std::chrono::steady_clock::now();
            timing.context_resume_ms+=std::chrono::duration<double,std::milli>(end-resume).count();
            timing.shared_wait_ms+=std::chrono::duration<double,std::milli>(end-begin).count();
            if(timing.shared_wait_calls<=timing.wait_each_ms.size()) {
                timing.wait_each_ms[timing.shared_wait_calls-1]=timing.wait_completion_ms-previous_wait;
                timing.resume_each_ms[timing.shared_wait_calls-1]=std::chrono::duration<double,std::milli>(end-resume).count();
            }
        }
    }
    ComPtr<ID3D11Texture2D> make_texture(DXGI_FORMAT format,int w,int h){
        D3D11_TEXTURE2D_DESC desc{};desc.Width=w;desc.Height=h;desc.MipLevels=1;desc.ArraySize=1;desc.Format=format;desc.SampleDesc.Count=1;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        ComPtr<ID3D11Texture2D> result;hr_check(device->CreateTexture2D(&desc,nullptr,&result),"Create owned GPU texture");return result;
    }
    ComPtr<ID3D11ShaderResourceView> source_view(ID3D11Texture2D* texture,DXGI_FORMAT format){
        for(auto& slot:output_pool)if(slot->texture.Get()==texture)return slot->views[(format==DXGI_FORMAT_R8_UNORM||format==DXGI_FORMAT_R16_UNORM)?0:1];
        D3D11_SHADER_RESOURCE_VIEW_DESC desc{};desc.Format=format;desc.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D;desc.Texture2D.MipLevels=1;
        ComPtr<ID3D11ShaderResourceView> view;hr_check(device->CreateShaderResourceView(texture,&desc,&view),"Create native Y/UV shader view");return view;
    }
    ComPtr<ID3D11RenderTargetView> target_view(ID3D11Texture2D* texture,DXGI_FORMAT format){
        for(auto& slot:output_pool)if(slot->texture.Get()==texture)return slot->targets[(format==DXGI_FORMAT_R8_UNORM||format==DXGI_FORMAT_R16_UNORM)?0:1];
        D3D11_RENDER_TARGET_VIEW_DESC desc{};desc.Format=format;desc.ViewDimension=D3D11_RTV_DIMENSION_TEXTURE2D;
        ComPtr<ID3D11RenderTargetView> view;hr_check(device->CreateRenderTargetView(texture,&desc,&view),"Create native Y/UV target view");return view;
    }
    GpuFrame make_output(int w,int h,int64_t pts,DXGI_FORMAT format=DXGI_FORMAT_NV12){
        for(auto& slot:output_pool)if(slot.use_count()==1&&slot->width==w&&slot->height==h&&slot->format==format)return {slot->texture,0,w,h,pts,slot};
        auto slot=std::make_shared<OutputSlot>();slot->width=w;slot->height=h;slot->format=format;slot->texture=make_texture(format,w,h);
        for(unsigned p=0;p<2;++p){auto viewFormat=plane_format(format,p);slot->views[p]=source_view(slot->texture.Get(),viewFormat);slot->targets[p]=target_view(slot->texture.Get(),viewFormat);}
        if(output_pool.size()<64)output_pool.push_back(slot);
        return {slot->texture,0,w,h,pts,slot};
    }
    // Reuse the source planes already owned by our surface pool. The local
    // COM references survive the scene wait; prepare takes its own references.
    // External/nonpooled textures keep the existing one-call view fallback.
    void prepare_rife(const GpuFrame& a,const GpuFrame& b,const MotionSynthesizer::GpuWait& wait,const MotionSynthesizer::PairReady& ready,const MotionSynthesizer::CpuWork& initialization){
        auto ta=shader_source_texture(a),tb=shader_source_texture(b);
        MotionSynthesizer::SourcePlanes planes{};
        std::array<ComPtr<ID3D11ShaderResourceView>,4> owned{};
        for(auto& slot:output_pool){
            if(slot->texture.Get()==ta.Get()){owned[0]=slot->views[0];owned[2]=slot->views[1];}
            if(slot->texture.Get()==tb.Get()){owned[1]=slot->views[0];owned[3]=slot->views[1];}
        }
        // Keep the fallback lazy: a cut/still pair must not create new views
        // before its gate has rejected inference.
        if(std::all_of(owned.begin(),owned.end(),[](const auto& view){return bool(view);}))
            for(size_t i=0;i<owned.size();++i)planes[i]=owned[i].Get();
        native_synthesizer->prepare(ta.Get(),tb.Get(),a.width,a.height,wait,ready,planes,initialization);
    }
    struct SceneAnalysis {SceneDecision decision;bool identical=false;};
    void queue_scene(const GpuFrame& a,const GpuFrame& b){
        auto ta=shader_source_texture(a),tb=shader_source_texture(b);
        const auto format=texture_format(a);
        auto va=source_view(ta.Get(),plane_format(format,0)),vb=source_view(tb.Get(),plane_format(format,0));
        auto ua=source_view(ta.Get(),plane_format(format,1)),ub=source_view(tb.Get(),plane_format(format,1));
        const UINT zero[4]={};context->ClearUnorderedAccessViewUint(scene_uav.Get(),zero);
        const uint32_t dimensions[4]={uint32_t(a.width),uint32_t(a.height),0,0};context->UpdateSubresource(scene_constants.Get(),0,nullptr,dimensions,0,0);
        ID3D11ShaderResourceView* views[]={va.Get(),vb.Get(),ua.Get(),ub.Get()};ID3D11UnorderedAccessView* uav=scene_uav.Get();ID3D11Buffer* constants=scene_constants.Get();
        context->CSSetShader(scene_compute.Get(),nullptr,0);context->CSSetConstantBuffers(0,1,&constants);context->CSSetShaderResources(0,4,views);context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);context->Dispatch(4,3,1);
        context->CSSetShader(scene_identity.Get(),nullptr,0);context->Dispatch(4096,1,1);
        ID3D11ShaderResourceView* empty[4]={};uav=nullptr;context->CSSetShaderResources(0,4,empty);context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);
        context->CSSetShader(scene_match.Get(),nullptr,0);
        ID3D11UnorderedAccessView* matchUav=scene_uav.Get();context->CSSetUnorderedAccessViews(0,1,&matchUav,nullptr);context->Dispatch(80,1,1);
        matchUav=nullptr;context->CSSetUnorderedAccessViews(0,1,&matchUav,nullptr);
        const D3D11_BOX aggregateBox{0,0,0,UINT(sizeof(SceneStats)+sizeof(uint32_t)),1,1};
        context->CopySubresourceRegion(scene_readback.Get(),0,0,0,0,scene_buffer.Get(),0,&aggregateBox);
    }
    SceneAnalysis read_scene(DeviceScope& scope,bool copy_completed=false){
        D3D11_MAPPED_SUBRESOURCE mapped{};
        const bool shared=completion_mode==GpuCompletionMode::context_ordered;
        HRESULT mappedResult=context->Map(scene_readback.Get(),0,D3D11_MAP_READ,shared&&!copy_completed?D3D11_MAP_FLAG_DO_NOT_WAIT:0,&mapped);
        if(shared && mappedResult==DXGI_ERROR_WAS_STILL_DRAWING) {
            wait_and_resume(scope);
            // The fence/query completed this private staging copy. Finish its
            // CPU mapping once; DO_NOT_WAIT can still report STILL_DRAWING
            // here and needlessly submit another full shared-context fence.
            mappedResult=context->Map(scene_readback.Get(),0,D3D11_MAP_READ,0,&mapped);
        }
        hr_check(mappedResult,"Read aggregate scene statistics");
        SceneStats statistics{};memcpy(&statistics,mapped.pData,sizeof(statistics));uint32_t difference=0;memcpy(&difference,static_cast<const uint8_t*>(mapped.pData)+sizeof(statistics),sizeof(difference));context->Unmap(scene_readback.Get(),0);return {scene_detector.evaluate(statistics),difference==0};
    }
    ComPtr<ID3D11Texture2D> shader_source_texture(const GpuFrame& frame){
        D3D11_TEXTURE2D_DESC desc{};frame.texture->GetDesc(&desc);
        if(desc.ArraySize==1 && desc.Width==UINT(frame.width) && desc.Height==UINT(frame.height) && (desc.BindFlags&D3D11_BIND_SHADER_RESOURCE))return frame.texture;
        // Decoder arrays or decode-only surfaces are first captured on GPU.
        // All draws share the immediate context, so copy ordering is preserved.
        auto result=make_texture(desc.Format,frame.width,frame.height);D3D11_BOX box{0,0,0,UINT(frame.width),UINT(frame.height),1};
        context->CopySubresourceRegion(result.Get(),0,0,0,0,frame.texture.Get(),frame.array_slice,&box);return result;
    }
    void draw_plane(ID3D11RenderTargetView* target,ID3D11ShaderResourceView* a,int w,int h){
        ID3D11ShaderResourceView* views[]={a,nullptr};
        context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);context->VSSetShader(vertex_shader.Get(),nullptr,0);context->GSSetShader(nullptr,nullptr,0);context->HSSetShader(nullptr,nullptr,0);context->DSSetShader(nullptr,nullptr,0);context->PSSetShader(p010_pixel_shader.Get(),nullptr,0);
        context->OMSetRenderTargets(1,&target,nullptr);context->OMSetBlendState(nullptr,nullptr,0xffffffff);context->OMSetDepthStencilState(nullptr,0);context->RSSetState(rasterizer.Get());
        D3D11_VIEWPORT viewport{0,0,float(w),float(h),0,1};context->RSSetViewports(1,&viewport);context->PSSetShaderResources(0,2,views);context->Draw(3,0);
        // D3D11 treats the two NV12 planes as the same resource for hazards.
        ID3D11ShaderResourceView* empty[]={nullptr,nullptr};context->PSSetShaderResources(0,2,empty);context->OMSetRenderTargets(0,nullptr,nullptr);
    }
    void normalize_p010(const GpuFrame& input,const GpuFrame& output) {
        if(!p010_pixel_shader) {
            hr_check(device->CreatePixelShader(shaders::p010::ps,sizeof(shaders::p010::ps),nullptr,&p010_pixel_shader),"Create P010 normalization shader");
        }
        if(!p010_capture || p010_width!=input.width || p010_height!=input.height) {
            D3D11_TEXTURE2D_DESC desc{};desc.Width=input.width;desc.Height=input.height;
            desc.MipLevels=1;desc.ArraySize=1;desc.Format=DXGI_FORMAT_P010;desc.SampleDesc.Count=1;
            desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            ComPtr<ID3D11Texture2D> capture;hr_check(device->CreateTexture2D(&desc,nullptr,&capture),"Create P010 capture texture");
            std::array<ComPtr<ID3D11ShaderResourceView>,2> views;
            views[0]=source_view(capture.Get(),DXGI_FORMAT_R16_UINT);
            views[1]=source_view(capture.Get(),DXGI_FORMAT_R16G16_UINT);
            p010_capture=std::move(capture);p010_views=std::move(views);
            p010_width=input.width;p010_height=input.height;
        }
        // Copy the selected decoder slice while its sample is retained. This
        // removes decoder padding without reading video pixels back to the CPU.
        D3D11_BOX box{0,0,0,UINT(input.width),UINT(input.height),1};
        context->CopySubresourceRegion(p010_capture.Get(),0,0,0,0,input.texture.Get(),input.array_slice,&box);
        for(unsigned plane=0;plane<2;++plane) {
            auto target=target_view(output.texture.Get(),plane?DXGI_FORMAT_R16G16_UNORM:DXGI_FORMAT_R16_UNORM);
            draw_plane(target.Get(),p010_views[plane].Get(),plane?input.width/2:input.width,plane?input.height/2:input.height);
        }
    }

};

GpuRifeEngine::GpuRifeEngine(const std::filesystem::path& runtime,ID3D11Device* device,ID3D11DeviceContext* context,HANDLE decoder_mutex,GpuCompletionMode mode):impl_(std::make_unique<Impl>(runtime,device,context,decoder_mutex,mode)){}
MotionAnalysisInfo GpuRifeEngine::analysis_info() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->native_synthesizer?impl_->native_synthesizer->analysis_info():MotionAnalysisInfo{};
}
GpuRifeEngine::~GpuRifeEngine()=default;
void GpuRifeEngine::enable_timing(bool enabled,bool rife_detail,unsigned interval){std::lock_guard<std::mutex> lock(impl_->mutex);impl_->trace_timing=enabled;impl_->trace_rife=enabled&&rife_detail;impl_->profile_interval=(std::max)(1u,interval);}
GpuPairTiming GpuRifeEngine::pair_timing()const{std::lock_guard<std::mutex> lock(impl_->mutex);return impl_->timing;}
std::string GpuRifeEngine::device_name()const{return impl_->name;}
bool GpuRifeEngine::queued_completion()const{return impl_->completion_mode==GpuCompletionMode::context_ordered;}
void GpuRifeEngine::set_wait_service(WaitService service){std::lock_guard<std::mutex> lock(impl_->mutex);impl_->wait_service=std::move(service);}
int GpuRifeEngine::startup_warmup(int width,int height,bool multiple){
    // A freshly reopened graph can overlap optional loading or retirement of
    // its predecessor. CUDA initialization here has no cancellation callback;
    // leave that contention to the interruptible real-pair preparation path.
    if(MotionSynthesizer::preload_pending() || MotionSynthesizer::optional_preload_pending() || retirement_pending())return -1;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    // Device resource creation and CUDA-only work do not modify the shared
    // immediate context. Do not hold its decoder/multithread lock here.
    return impl_->native_synthesizer?impl_->native_synthesizer->startup_warmup(width,height,multiple):-1;
}
bool GpuRifeEngine::startup_ready()const{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->native_synthesizer&&impl_->native_synthesizer->startup_ready();
}
bool GpuRifeEngine::retire_during_preparation(std::unique_ptr<GpuRifeEngine>& engine,std::shared_ptr<void> host_lifetime)noexcept{
    if(!engine||!MotionSynthesizer::optional_preload_pending())return false;
    if(retired_engines.fetch_add(1)>=2){--retired_engines;return false;}
    HMODULE module=nullptr;
    try{
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,reinterpret_cast<LPCWSTR>(&retire_engine_entry),&module)){
            --retired_engines;return false;
        }
        auto job=std::make_unique<RetiredEngine>();job->module=module;
        job->host_lifetime=std::move(host_lifetime);job->engine=std::move(engine);
        HANDLE thread=CreateThread(nullptr,0,retire_engine_entry,job.get(),0,nullptr);
        if(!thread){engine=std::move(job->engine);FreeLibrary(module);--retired_engines;return false;}
        job.release();CloseHandle(thread);return true;
    }catch(...){if(module)FreeLibrary(module);--retired_engines;return false;}
}
unsigned GpuRifeEngine::retirement_pending()noexcept{return retired_engines.load();}
void GpuRifeEngine::reset()noexcept{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->scene_detector.reset();impl_->scene_previous=nullptr;impl_->scene_previous_pts=-1;
    try{
        Impl::DeviceScope scope(*impl_);
        if(impl_->native_synthesizer) {
            impl_->native_synthesizer->invalidate_history();
            impl_->cached_texture.Reset();impl_->cached_lease.reset();return;
        }
    }catch(...){OutputDebugStringA("RIFE: shared D3D11 lock unavailable during history reset\n");}
}
GpuFrame GpuRifeEngine::copy(const GpuFrame& input,bool submit_now){
    std::lock_guard<std::mutex> lock(impl_->mutex);impl_->validate(input,true);Impl::DeviceScope device(*impl_);
    D3D11_TEXTURE2D_DESC desc{};input.texture->GetDesc(&desc);
    auto result=impl_->make_output(input.width,input.height,input.pts,desc.Format);
    result.hdr=input.hdr;result.hdr_transfer=input.hdr_transfer;
    if(desc.Format==DXGI_FORMAT_P010)impl_->normalize_p010(input,result);
    else {
        D3D11_BOX box{0,0,0,UINT(input.width),UINT(input.height),1};
        impl_->context->CopySubresourceRegion(result.texture.Get(),0,0,0,0,input.texture.Get(),input.array_slice,&box);
    }
    if(!queued_completion())impl_->wait_gpu(&device);
    else if(submit_now)impl_->context->Flush();
    return result;
}
PhaseBatch<GpuFrame> GpuRifeEngine::interpolate_pair(const GpuFrame& previous,const GpuFrame& current,const std::vector<int64_t>& timestamps,const std::atomic<bool>* cancelled){
    std::lock_guard<std::mutex> lock(impl_->mutex);impl_->validate(previous);impl_->validate(current);
    if(cancelled&&cancelled->load(std::memory_order_acquire))throw GpuWorkCancelled{};
    using Clock=std::chrono::steady_clock;
    auto tick=impl_->trace_timing?Clock::now():Clock::time_point{};
    impl_->timing={};
    auto mark=[&](double& stage){if(impl_->trace_timing){auto now=Clock::now();stage=std::chrono::duration<double,std::milli>(now-tick).count();tick=now;}};
    if(texture_format(previous)!=texture_format(current)||previous.width!=current.width||previous.height!=current.height||previous.pts<0||current.pts<=previous.pts||current.pts-previous.pts>100000000LL)throw std::invalid_argument("GPU phase interpolation requires matching adjacent frames");
    if(timestamps.size()>32)throw std::invalid_argument("More than 32 motion phases per source pair are unsupported");
    int64_t last=previous.pts;for(auto pts:timestamps){if(pts<=last||pts>=current.pts)throw std::invalid_argument("Motion timestamps must increase inside source pair");last=pts;}
    PhaseBatch<GpuFrame> result{};if(timestamps.empty())return result;
    if(impl_->native_synthesizer) {
        // A nominal x2 clock can land away from the midpoint when container
        // duration and actual PTS differ, or when input is VFR. Synthesize the
        // requested time; never label a midpoint as a different phase.
        if(HANDLE ready=impl_->native_synthesizer->startup_warmup_event()) {
            if(impl_->wait_for_completion(ready,nullptr)!=WAIT_OBJECT_0)
                throw std::runtime_error("Startup CUDA warmup timeout");
        }
        Impl::DeviceScope scope(*impl_);
        mark(impl_->timing.context_wait_ms);
        MotionSynthesizer::CpuWork initialization;
        if(impl_->native_synthesizer->initialization_pending())initialization=[&](const std::function<void()>& work){
            scope.release();
            try{
                // The loader owns its engine/context independently of this
                // pair. Leave that uninterruptible driver work on its thread.
                while(cancelled&&(MotionSynthesizer::preload_pending()||retirement_pending())){
                    if(cancelled->load(std::memory_order_acquire))throw GpuWorkCancelled{};
                    Sleep(1);
                }
                if(cancelled&&cancelled->load(std::memory_order_acquire))throw GpuWorkCancelled{};
                work();
            }catch(...){scope.acquire();throw;}
            scope.acquire();
        };
        const bool profile=impl_->trace_rife&&(++impl_->profile_sequence%impl_->profile_interval==0);
        impl_->native_synthesizer->begin_timing(profile,current.pts);
        try {
            const bool continuous=impl_->cached_texture.Get()==previous.texture.Get()&&impl_->cached_pts==previous.pts&&
               impl_->cached_slice==previous.array_slice;
            if(!continuous)impl_->native_synthesizer->invalidate_history();
            impl_->native_synthesizer->set_hdr_transfer(previous.hdr_transfer);
            // Scene history tracks adjacency independently of the RIFE cache, which
            // is intentionally discarded at cuts and identical pairs.
            if(impl_->scene_previous!=previous.texture.Get()||impl_->scene_previous_pts!=previous.pts||impl_->scene_previous_slice!=previous.array_slice)impl_->scene_detector.reset();
            impl_->queue_scene(previous,current);
            Impl::SceneAnalysis analysis{};
            // Finish this pair's scene readback before preparing the RIFE guide.
            // Release shared device locks while waiting and service ready outputs.
            // The gate uses the current pair, including after seeks and cuts.
            const bool combined=queued_completion();
            if(combined) {
                if(impl_->trace_timing)++impl_->timing.combined_scene_pairs;
                mark(impl_->timing.scene_ms);
                impl_->prepare_rife(previous,current,
                    [&]{impl_->wait_and_resume(scope);},[&]{
                        analysis=impl_->read_scene(scope,true);
                        return !analysis.decision.cut&&!analysis.identical;
                    },initialization);
                // The scene-gate wait is charged to preparation. These
                // stage timings measure CPU wall time, not isolated GPU work.
                mark(impl_->timing.prepare_ms);
            }else {
                analysis=impl_->read_scene(scope);
                mark(impl_->timing.scene_ms);
            }
            impl_->scene_previous=current.texture.Get();impl_->scene_previous_pts=current.pts;impl_->scene_previous_slice=current.array_slice;
            if(analysis.decision.cut) {
                impl_->native_synthesizer->invalidate_history();
                result.quality.scene_cut=true;impl_->cached_texture.Reset();impl_->cached_lease.reset();if(profile)impl_->timing.rife=snapshot_rife_timing(impl_->native_synthesizer->timing());return result;
            }
            if(analysis.identical) {
                for(auto pts:timestamps) {auto exact=previous;exact.pts=pts;result.frames.push_back(std::move(exact));}
                result.quality.identical_warp_skipped=true;impl_->native_synthesizer->invalidate_history();
            }else{
                if(!combined) {
                    MotionSynthesizer::GpuWait wait;
                    if(queued_completion())wait=[&]{impl_->wait_and_resume(scope);};
                    impl_->prepare_rife(previous,current,wait,{},initialization);
                    mark(impl_->timing.prepare_ms);
                }
                result.quality.repetition_known=true;
                std::vector<MotionSynthesizer::PhaseTarget> targets;
                std::vector<ComPtr<ID3D11RenderTargetView>> targetViews;
                targets.reserve(timestamps.size());targetViews.reserve(timestamps.size()*2);
                const auto allocationTick=impl_->trace_timing?Clock::now():Clock::time_point{};
                for(size_t i=0;i<timestamps.size();++i) {
                    if(!impl_->native_synthesizer->prepared()) {
                        auto held=previous;held.pts=timestamps[i];result.frames.push_back(std::move(held));
                        result.quality.repeated_mask|=uint32_t(1)<<i;continue;
                    }
                    auto output=impl_->make_output(previous.width,previous.height,timestamps[i],texture_format(previous));
                    output.hdr=previous.hdr;output.hdr_transfer=previous.hdr_transfer;
                    auto y=impl_->target_view(output.texture.Get(),plane_format(texture_format(previous),0));
                    auto uv=impl_->target_view(output.texture.Get(),plane_format(texture_format(previous),1));
                    const float fraction=(std::min)(std::nextafter(1.0f,0.0f),float(double(timestamps[i]-previous.pts)/double(current.pts-previous.pts)));
                    targets.push_back({y.Get(),uv.Get(),fraction});
                    targetViews.push_back(std::move(y));targetViews.push_back(std::move(uv));
                    result.frames.push_back(std::move(output));result.quality.native_synthesized_mask|=uint32_t(1)<<i;
                }
                if(impl_->trace_timing)impl_->timing.target_allocation_ms=std::chrono::duration<double,std::milli>(Clock::now()-allocationTick).count();
                if(!targets.empty()) {
                    MotionSynthesizer::CpuWork optional_setup;
                    if(impl_->native_synthesizer->optional_setup_pending(targets.size()))optional_setup=[&](const std::function<void()>& work){
                        scope.release();
                        try{work();}catch(...){scope.acquire();throw;}
                        scope.acquire();
                    };
                    impl_->native_synthesizer->render_phases(targets.data(),targets.size(),[&](HANDLE event){impl_->wait_cuda_ready(scope,event);},optional_setup,cancelled);
                }
            }
            mark(impl_->timing.submit_ms);
            impl_->cached_texture=current.texture;impl_->cached_lease=current.lease;
            impl_->cached_pts=current.pts;impl_->cached_slice=current.array_slice;
            if(!queued_completion())impl_->wait_gpu(&scope);
            else if(result.quality.native_synthesized_mask) {
                // Submit the output batch before downstream can
                // park this thread waiting for a renderer sample/presentation.
                // Flush is asynchronous: retain the shared-context completion
                // contract and the existing RIFE resources, with no new GPU wait.
                impl_->context->Flush();
                if(impl_->trace_timing)++impl_->timing.queued_output_flushes;
            }
            mark(impl_->timing.completion_ms);
            if(profile)impl_->timing.rife=snapshot_rife_timing(impl_->native_synthesizer->timing());
            return result;
        }catch(const GpuWorkCancelled&){
            // render_phases drained CUDA and restored graphics ownership. Keep
            // compatible allocations; cancellation is not an execution failure.
            impl_->scene_detector.reset();impl_->scene_previous=nullptr;
            impl_->cached_texture.Reset();impl_->cached_lease.reset();
            impl_->native_synthesizer->invalidate_history();throw;
        }catch(...){impl_->scene_detector.reset();impl_->scene_previous=nullptr;if(!scope.released)impl_->native_synthesizer->reset();throw;}
    }
    throw std::logic_error("Native synthesizer unavailable");
}

}
