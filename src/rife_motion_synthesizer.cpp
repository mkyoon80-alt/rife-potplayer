#include "nvof/motion_synthesizer.hpp"
#include "nvof/rife_optimization_policy.hpp"
#include "rife_shaders.hpp"
#include "rife_packed_shaders.hpp"
#include "nvof/playback_diagnostics.hpp"
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstring>
#include <cuda_d3d11_interop.h>
#include <wrl/client.h>
#include <dxgi.h>
#include <d3d11_4.h>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <map>
#include <set>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <chrono>

namespace nvof {
namespace {
using Microsoft::WRL::ComPtr;
namespace sh=shaders::rife;
constexpr unsigned GW=960,GH=540,N=GW*GH,PhaseCapacity=4;
std::atomic<uint64_t> contexts_created{0};
std::atomic<uint64_t> handoffs{0};
std::atomic<uint64_t> loads{0},inferences{0},graph_launches{0},graph_builds{0},graph_fallbacks{0};
bool environment_flag(const wchar_t* name,bool fallback=false){wchar_t value[8]{};const auto length=GetEnvironmentVariableW(name,value,8);return length>0?value[0]==L'1':fallback;}
#if defined(RIFE_ENABLE_E03_E04_BY_DEFAULT)
constexpr bool default_e03_e04=true;
#else
constexpr bool default_e03_e04=false;
#endif
void ck(HRESULT result,const char* action){if(FAILED(result))throw std::runtime_error(std::string(action)+" HRESULT="+std::to_string(unsigned(result)));}
struct Library {
    HMODULE handle=nullptr;
    explicit Library(const std::filesystem::path& path){
        handle=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if(!handle)throw std::runtime_error("RIFE runtime DLL unavailable: "+path.filename().string()+" Win32="+std::to_string(GetLastError()));
    }
    ~Library(){if(handle)FreeLibrary(handle);}
    template<class F> F get(const char* name){auto f=GetProcAddress(handle,name);if(!f)throw std::runtime_error(std::string("RIFE runtime symbol unavailable: ")+name);return reinterpret_cast<F>(f);}
};
std::filesystem::path system_cuda_driver(){
    wchar_t path[MAX_PATH]{};
    if(!GetSystemDirectoryW(path,MAX_PATH))throw std::runtime_error("Cannot locate CUDA driver");
    return std::filesystem::path(path)/L"nvcuda.dll";
}
struct Cuda {
    Library library;
    Library driver{system_cuda_driver()};
    decltype(&::cuMemsetD32Async) fill32=driver.get<decltype(&::cuMemsetD32Async)>("cuMemsetD32Async");
    void fill_phase(void* dst,float phase,cudaStream_t stream){
        unsigned bits=0;static_assert(sizeof(bits)==sizeof(phase));std::memcpy(&bits,&phase,sizeof(bits));
        auto status=fill32(reinterpret_cast<CUdeviceptr>(dst),bits,N,reinterpret_cast<CUstream>(stream));
        if(status!=CUDA_SUCCESS)throw std::runtime_error("RIFE CUDA phase fill failed: "+std::to_string(int(status)));
    }
#define CUDA_FN(name) decltype(&::name) name=library.get<decltype(&::name)>(#name)
    CUDA_FN(cudaImportExternalMemory); CUDA_FN(cudaExternalMemoryGetMappedBuffer); CUDA_FN(cudaDestroyExternalMemory); CUDA_FN(cudaFree); CUDA_FN(cudaMalloc);
    CUDA_FN(cudaImportExternalSemaphore); CUDA_FN(cudaDestroyExternalSemaphore); CUDA_FN(cudaWaitExternalSemaphoresAsync); CUDA_FN(cudaSignalExternalSemaphoresAsync);
    CUDA_FN(cudaSetDevice); CUDA_FN(cudaD3D11GetDevice);
    CUDA_FN(cudaGraphicsD3D11RegisterResource); CUDA_FN(cudaGraphicsUnregisterResource);
    CUDA_FN(cudaGraphicsMapResources); CUDA_FN(cudaGraphicsUnmapResources);
    CUDA_FN(cudaGraphicsResourceGetMappedPointer); CUDA_FN(cudaGraphicsResourceSetMapFlags);
    CUDA_FN(cudaStreamCreateWithFlags); CUDA_FN(cudaStreamDestroy); CUDA_FN(cudaStreamSynchronize);
    CUDA_FN(cudaGetErrorString); CUDA_FN(cudaMemcpyAsync);
    CUDA_FN(cudaStreamBeginCapture); CUDA_FN(cudaStreamEndCapture);
    CUDA_FN(cudaGraphInstantiateWithFlags); CUDA_FN(cudaGraphLaunch); CUDA_FN(cudaGraphDestroy); CUDA_FN(cudaGraphExecDestroy);
    CUDA_FN(cudaEventQuery); CUDA_FN(cudaStreamQuery);
    CUDA_FN(cudaEventCreateWithFlags); CUDA_FN(cudaEventRecord); CUDA_FN(cudaEventElapsedTime); CUDA_FN(cudaEventDestroy);
#undef CUDA_FN
    explicit Cuda(const std::filesystem::path& dir):library(dir/L"cudart64_13.dll"){}
    void check(cudaError_t e,const char* action){if(e!=cudaSuccess)throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(e));}
};
#include "rife_external_buffers.inl"
#include "rife_profiler.inl"
#include "rife_graph.inl"
struct Logger: nvinfer1::ILogger {
    std::mutex mutex;
    std::string error;
    std::string message(){std::lock_guard<std::mutex> lock(mutex);return error;}
    void log(Severity severity,const char* msg) noexcept override {
        if(severity>Severity::kWARNING)return;
        try{std::lock_guard<std::mutex> lock(mutex);if(severity<=Severity::kERROR)error=msg;OutputDebugStringA((std::string("RIFE TensorRT: ")+msg+"\n").c_str());}catch(...){}
    }
};
std::shared_ptr<Logger> common_logger(){
    // TensorRT uses one logger for the loaded library. Keep it alive until
    // every cached engine/runtime has been destroyed, including module unload.
    static auto logger=std::make_shared<Logger>();return logger;
}
struct SharedEngine {
    std::shared_ptr<Cuda> cuda;
    Library library;
    std::shared_ptr<Logger> logger=common_logger();
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    int device;
    // One idle context per engine/GPU. No image, graph or D3D resource is cached.
    std::mutex context_mutex;
    std::unique_ptr<nvinfer1::IExecutionContext> idle_context;
    std::unique_ptr<nvinfer1::IExecutionContext> make_context(){
        auto result=std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if(!result)throw std::runtime_error("Cannot allocate RIFE execution context");
        ++contexts_created;return result;
    }
    void prewarm_context(){
        cuda->check(cuda->cudaSetDevice(device),"Select preload GPU");
        std::lock_guard<std::mutex> lock(context_mutex);if(!idle_context)idle_context=make_context();
    }
    std::unique_ptr<nvinfer1::IExecutionContext> take_context(){
        if(environment_flag(L"RIFE_DISABLE_CONTEXT_REUSE"))return make_context();
        std::lock_guard<std::mutex> lock(context_mutex);
        if(idle_context)return std::move(idle_context);
        return make_context();
    }
    void recycle_context(std::unique_ptr<nvinfer1::IExecutionContext>& value)noexcept{
        if(!value||environment_flag(L"RIFE_DISABLE_CONTEXT_REUSE"))return;
        // Caller drained its stream and destroyed its graph. Do not retain
        // addresses of the buffers that the caller is about to release.
        if(!value->setTensorAddress("input",nullptr)||!value->setTensorAddress("motion",nullptr))return;
        try{std::lock_guard<std::mutex> lock(context_mutex);if(!idle_context)idle_context=std::move(value);}catch(...){}
    }
    SharedEngine(const std::filesystem::path& dir,const std::filesystem::path& plan,std::shared_ptr<Cuda> cu,int gpu):cuda(std::move(cu)),library(dir/L"nvinfer_10.dll"),device(gpu){
        cuda->check(cuda->cudaSetDevice(device),"Select inference GPU");
        auto version=library.get<int32_t(*)()>("getInferLibVersion");
        if(version()!=NV_TENSORRT_VERSION)throw std::runtime_error("TensorRT version differs from the bundled build. Run Prepare-Engine.ps1 during setup.");
        using Create=void*(*)(void*,int32_t);
        runtime.reset(static_cast<nvinfer1::IRuntime*>(library.get<Create>("createInferRuntime_INTERNAL")(logger.get(),NV_TENSORRT_VERSION)));
        if(!runtime)throw std::runtime_error("TensorRT runtime creation failed: "+logger->message());
        std::ifstream file(plan,std::ios::binary|std::ios::ate);
        if(!file)throw std::runtime_error("RIFE engine is not prepared. Run Prepare-Engine.ps1 before playback.");
        const auto bytes=file.tellg();
        if(bytes<1024||bytes>1024ll*1024*1024)throw std::runtime_error("Invalid RIFE engine file");
        std::vector<char> data(static_cast<size_t>(bytes));file.seekg(0);file.read(data.data(),data.size());
        if(!file)throw std::runtime_error("Cannot read RIFE engine");
        engine.reset(runtime->deserializeCudaEngine(data.data(),data.size()));
        if(!engine)throw std::runtime_error("RIFE engine is incompatible or damaged. Run Prepare-Engine.ps1 for this computer before playback. "+logger->message());
        const auto a=engine->getTensorShape("input"),b=engine->getTensorShape("motion");
        if(engine->getNbIOTensors()!=2 || a.nbDims!=4 || b.nbDims!=4 || a.d[0]!=1 || a.d[1]!=7 || a.d[2]!=GH || a.d[3]!=GW || b.d[0]!=1 || b.d[1]!=5 || b.d[2]!=GH || b.d[3]!=GW || engine->getTensorDataType("input")!=nvinfer1::DataType::kFLOAT || engine->getTensorDataType("motion")!=nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("RIFE engine tensor contract mismatch (expected Standard 960 motion/mask engine)");
        ++loads;
    }
    ~SharedEngine(){cuda->cudaSetDevice(device);idle_context.reset();engine.reset();runtime.reset();}
};
std::shared_ptr<SharedEngine> cached_adapter_engine(const std::filesystem::path& dir,IDXGIAdapter* adapter,int& device){
    // Strong references keep the plan alive across file changes in this module.
    // Each synthesizer owns a separate execution context and CUDA stream.
    static std::mutex mutex;
    static std::map<std::wstring,std::shared_ptr<Cuda>> runtimes;
    static std::map<std::wstring,std::shared_ptr<SharedEngine>> engines;
    std::lock_guard<std::mutex> lock(mutex);
    // Preserve the package root even when runtime/ is a directory junction.
    // The model cache belongs beside the caller's runtime, not its DLL target.
    auto normalized=std::filesystem::absolute(dir).lexically_normal();
    auto& cu=runtimes[normalized.wstring()];if(!cu)cu=std::make_shared<Cuda>(normalized);
    cu->check(cu->cudaD3D11GetDevice(&device,adapter),"Decoder GPU must support CUDA/D3D11 interop");
    auto plan=normalized.parent_path()/L"engines"/L"rife425_standard_960_fp16.engine";
    const auto key=plan.wstring()+L"/"+std::to_wstring(device);
    auto& result=engines[key];if(!result)result=std::make_shared<SharedEngine>(normalized,plan,cu,device);
    return result;
}
std::shared_ptr<SharedEngine> cached_engine(const std::filesystem::path& dir,ID3D11Device* d,int& device){
    ComPtr<IDXGIDevice> dxgi;ComPtr<IDXGIAdapter> adapter;
    ck(d->QueryInterface(IID_PPV_ARGS(&dxgi)),"Get inference DXGI device");ck(dxgi->GetAdapter(&adapter),"Get inference adapter");
    return cached_adapter_engine(dir,adapter.Get(),device);
}
#include "rife_pair_head_cache.inl"
#include "rife_packed_motion.inl"
#include "rife_preload.inl"
std::filesystem::path default_runtime(){
    HMODULE module=nullptr;GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(&default_runtime),&module);
    wchar_t path[32768]{};GetModuleFileNameW(module,path,32768);return std::filesystem::path(path).parent_path()/L"runtime";
}
}

struct MotionSynthesizer::Impl {
    ComPtr<ID3D11Device> d;ComPtr<ID3D11DeviceContext> c;
    std::filesystem::path directory;
    std::shared_ptr<SharedEngine> shared;
    std::unique_ptr<nvinfer1::IExecutionContext> execution;
    int gpu=-1;
    cudaStream_t stream=nullptr;
    std::array<cudaGraphicsResource*,2> resources{};
    ComPtr<ID3D11Buffer> input,motion,params;
    ComPtr<ID3D11UnorderedAccessView> inputUav;
    ComPtr<ID3D11ShaderResourceView> motionSrv;
    std::array<ComPtr<ID3D11ShaderResourceView>,PhaseCapacity> batchSrvs;
    std::array<ComPtr<ID3D11ShaderResourceView>,4> views;
    ComPtr<ID3D11ComputeShader> prepCS,timeCS;
    ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> pixel,pixelY,pixelUV;
    ComPtr<ID3D11SamplerState> sampler;ComPtr<ID3D11RasterizerState> raster;
    struct Params {
        float w=0,h=0;unsigned gw=GW,gh=GH;
        float phase=.5f;unsigned plane=0,tenBit=0,hdrTransfer=0;
        float aw=0,ah=0,ox=0,oy=0;
        unsigned portrait=0;float unused[3]{};
    } p;
    bool prepared=false;
    bool startup_warmed=false,startup_warm_pending=false;
    bool startup_head_required=false,startup_head_warmed=false;
    uint64_t startup_warm_value=0;
    float input_phase=.5f;
    RifeExternalBuffers external;
    const CompletionWait* completion_wait=nullptr;
    const CpuWork* startup_cuda_work=nullptr;
    bool external_requested=!environment_flag(L"RIFE_DISABLE_EXTERNAL_BUFFER_FENCES");
    RifeProfiler profiler;
    RifeGraph graph;
    bool batch_enabled=!environment_flag(L"RIFE_DISABLE_BATCH_HANDOFF");
    bool elide_final_batch_copy=environment_flag(L"RIFE_EXPERIMENT_FINAL_MOTION_COPY")&&!environment_flag(L"RIFE_REFERENCE_FINAL_MOTION_COPY");
    bool pair_head_requested=environment_flag(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",default_e03_e04);
    bool pair_head_unavailable=false;
    std::unique_ptr<RifePairHeadCache> pair_head;
    std::shared_ptr<RifeHeadPreparation> head_preparation;
    bool graph_enabled=!environment_flag(L"RIFE_DISABLE_CUDA_GRAPH");
    bool reference_interop=environment_flag(L"RIFE_REFERENCE_INTEROP");
    bool reference_composite=environment_flag(L"RIFE_REFERENCE_COMPOSITE");
    bool packed_requested=environment_flag(L"RIFE_EXPERIMENT_PACKED_MOTION",default_e03_e04);
    bool packed_selected=false;
    std::unique_ptr<RifePackedMotion> packed_motion;
    bool force_cuda_sync=environment_flag(L"RIFE_FORCE_CUDA_SYNC");
    Impl(ID3D11Device* device,ID3D11DeviceContext* context,const std::filesystem::path& dir):d(device),c(context),directory(dir.empty()?default_runtime():dir),profiler(device,context){
        if(!d||!c)throw std::invalid_argument("RIFE requires a D3D11 device/context");
    }
    void buffer(ComPtr<ID3D11Buffer>& b,unsigned bytes,unsigned bind,unsigned misc=0){D3D11_BUFFER_DESC desc{};desc.ByteWidth=bytes;desc.BindFlags=bind;desc.MiscFlags=misc;ck(d->CreateBuffer(&desc,nullptr,&b),"RIFE buffer");}
    void configure_resolution(int width,int height){
        const bool selected=packed_requested&&!reference_composite&&rife_packed_motion_resolution(width,height);
        if(selected==packed_selected)return;
        packed_selected=selected;
        packed_motion.reset();
        if(selected&&execution&&motionSrv)
            packed_motion=std::make_unique<RifePackedMotion>(d.Get(),c.Get(),motion.Get(),motionSrv.Get(),batchSrvs);
    }
    void initialize(const CpuWork& initialization={}){
        if(execution)return;
        auto initTick=RifeProfiler::Clock::now();
        auto initMark=[&](double& value){const auto now=RifeProfiler::Clock::now();value=std::chrono::duration<double,std::milli>(now-initTick).count();initTick=now;};
        try {
            auto initialize_cuda=[&]{
            shared=cached_engine(directory,d.Get(),gpu);auto& cu=*shared->cuda;cu.check(cu.cudaSetDevice(gpu),"Select RIFE GPU");
            initMark(profiler.data.init_engine_ms);
            auto contextTick=RifeProfiler::Clock::now();
            execution=shared->take_context();
            profiler.data.init_take_context_ms=RifeProfiler::ms(contextTick);
            if(!execution)throw std::runtime_error("Cannot allocate RIFE execution context");
            contextTick=RifeProfiler::Clock::now();
            cu.check(cu.cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"RIFE CUDA stream");
            profiler.data.init_stream_ms=RifeProfiler::ms(contextTick);
            profiler.attach(cu,stream);initMark(profiler.data.init_context_ms);
            };
            // Context/module loading can outlive graph connection. No immediate
            // context commands occur here; keep existing originals displayable.
            if(initialization)initialization(initialize_cuda);else initialize_cuda();
            auto& cu=*shared->cuda;
            auto create_buffers=[&](unsigned misc){buffer(input,N*7*4,D3D11_BIND_UNORDERED_ACCESS,misc);buffer(motion,N*5*4*(batch_enabled?PhaseCapacity+1:1),D3D11_BIND_SHADER_RESOURCE,misc);};
            if(external_requested){
                try{create_buffers(D3D11_RESOURCE_MISC_SHARED);external.initialize(cu,d.Get(),c.Get(),input.Get(),motion.Get());}
                catch(const std::exception& error){if(environment_flag(L"RIFE_REQUIRE_EXTERNAL_BUFFER_FENCES"))throw;external.release(cu);input.Reset();motion.Reset();OutputDebugStringA((std::string("RIFE explicit interop unavailable; using map/unmap: ")+error.what()+"\n").c_str());}
            }
            if(!external.active)create_buffers(0);
            buffer(params,sizeof(p),D3D11_BIND_CONSTANT_BUFFER);
            D3D11_UNORDERED_ACCESS_VIEW_DESC u{};u.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;u.Format=DXGI_FORMAT_R32_FLOAT;u.Buffer.NumElements=N*7;
            ck(d->CreateUnorderedAccessView(input.Get(),&u,&inputUav),"RIFE input UAV");
            D3D11_SHADER_RESOURCE_VIEW_DESC v{};v.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;v.Format=DXGI_FORMAT_R32_FLOAT;v.Buffer.NumElements=N*5;
            ck(d->CreateShaderResourceView(motion.Get(),&v,&motionSrv),"RIFE motion SRV");
            if(batch_enabled)for(unsigned i=0;i<PhaseCapacity;++i){
                v.Buffer.FirstElement=N*5*(i+1);
                ck(d->CreateShaderResourceView(motion.Get(),&v,&batchSrvs[i]),"RIFE batched motion SRV");
            }
            if(!external.active){
            cu.check(cu.cudaGraphicsD3D11RegisterResource(&resources[0],input.Get(),cudaGraphicsRegisterFlagsNone),"Register RIFE input");
            cu.check(cu.cudaGraphicsD3D11RegisterResource(&resources[1],motion.Get(),cudaGraphicsRegisterFlagsNone),"Register RIFE motion");
            if(!reference_interop&&!batch_enabled){
                // TensorRT only reads input and overwrites every output element.
                cu.check(cu.cudaGraphicsResourceSetMapFlags(resources[0],cudaGraphicsMapFlagsReadOnly),"RIFE input read-only mapping");
                cu.check(cu.cudaGraphicsResourceSetMapFlags(resources[1],cudaGraphicsMapFlagsWriteDiscard),"RIFE output discard mapping");
            }
            }
            initMark(profiler.data.init_buffers_ms);
            ck(d->CreateComputeShader(sh::prepareInput,sizeof(sh::prepareInput),nullptr,&prepCS),"RIFE input shader");
            ck(d->CreateComputeShader(sh::setTime,sizeof(sh::setTime),nullptr,&timeCS),"RIFE time shader");
            ck(d->CreateVertexShader(sh::vs,sizeof(sh::vs),nullptr,&vertex),"RIFE vertex shader");
            ck(d->CreatePixelShader(sh::composite,sizeof(sh::composite),nullptr,&pixel),"RIFE composite shader");
            ck(d->CreatePixelShader(sh::compositeY,sizeof(sh::compositeY),nullptr,&pixelY),"RIFE luma shader");
            ck(d->CreatePixelShader(sh::compositeUV,sizeof(sh::compositeUV),nullptr,&pixelUV),"RIFE chroma shader");
            if(packed_selected){
                const auto packedTick=RifeProfiler::Clock::now();
                packed_motion=std::make_unique<RifePackedMotion>(d.Get(),c.Get(),motion.Get(),motionSrv.Get(),batchSrvs);
                profiler.data.init_packed_ms=RifeProfiler::ms(packedTick);
            }
            D3D11_SAMPLER_DESC sd{};sd.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;sd.AddressU=sd.AddressV=sd.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;sd.MaxLOD=D3D11_FLOAT32_MAX;
            ck(d->CreateSamplerState(&sd,&sampler),"RIFE sampler");
            D3D11_RASTERIZER_DESC rd{};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;rd.DepthClipEnable=TRUE;
            ck(d->CreateRasterizerState(&rd,&raster),"RIFE raster state");initMark(profiler.data.init_shaders_ms);
        }catch(...){reset();throw;}
    }
    int startup_warmup(int width,int height,bool multiple_outputs){
        // Never wait for the background engine/context loader on the video
        // worker. It may continue while originals and audio play normally.
        if(!execution && preload_jobs.load()!=0)return -1;
        if(width>0&&height>0)configure_resolution(width,height);
        initialize();
        // A not-yet-loaded optimization must not extend the first-output
        // critical path. Warm cached split plans if available; otherwise use
        // the ready full engine and retain normal asynchronous E03 adoption.
        std::shared_ptr<RifeHeadPlans> startup_plans;
        if(multiple_outputs&&batch_enabled&&pair_head_requested&&!pair_head_unavailable){
            startup_plans=ready_head_plans(shared,directory);
            if(!startup_plans&&environment_flag(L"RIFE_HEAD_CACHE_SYNC_PREPARE"))
                startup_plans=cached_head_plans(shared,directory); // Deterministic probes only.
        }
        startup_head_required=bool(startup_plans);
        if(!external.active){startup_warmed=true;startup_head_required=false;return 0;}
        if(startup_ready())return 0;
        if(startup_warm_pending && external.fence->GetCompletedValue()<startup_warm_value)return -1;
        auto& cu=*shared->cuda;cu.check(cu.cudaSetDevice(gpu),"Select startup warmup GPU");
        // Fresh buffers have never been submitted to D3D. CUDA is their first
        // owner. The real guide later overwrites all input channels; no dummy
        // pixels or motion are ever exposed to the renderer or scene history.
        int enqueues=0;
        if(!startup_warmed){
            if(cu.fill32(reinterpret_cast<CUdeviceptr>(external.pointers[0]),0,N*7,reinterpret_cast<CUstream>(stream))!=CUDA_SUCCESS)
                throw std::runtime_error("Initialize startup warmup input");
            cu.fill_phase(static_cast<char*>(external.pointers[0])+N*6*4,.5f,stream);
            graph.run(cu,*execution,stream,external.pointers,graph_enabled);
            graph.run(cu,*execution,stream,external.pointers,graph_enabled);
            startup_warmed=true;enqueues=2;
        }
        if(startup_head_required&&!startup_head_warmed&&!pair_head_unavailable){
            // Readiness is polled; neither original delivery nor graph Run
            // waits for optional plan loading. The preload may already own it.
            if(startup_plans){
                std::unique_ptr<RifePairHeadCache> next;
                try{
                    next=std::make_unique<RifePairHeadCache>(std::move(startup_plans));next->initialize();
                    // Warm exactly the addresses used by real playback. This
                    // does not count dummy features as a real source pair.
                    RifePairHeadCache::HeadAdapter adapter{*next->contexts.head};
                    for(int i=0;i<2;++i){
                        next->head_graph.run(cu,adapter,stream,{external.pointers[0],next->features},graph_enabled);
                        next->tail_graph.run(cu,*next->contexts.tail,stream,external.pointers,graph_enabled);
                    }
                    pair_head=std::move(next);startup_head_warmed=true;enqueues+=4;
                }catch(const std::exception& error){
                    // Optional E03 failure retains the original full engine.
                    // Submitted work must finish before its buffers disappear.
                    cu.check(cu.cudaStreamSynchronize(stream),"Drain optional startup warmup");
                    next.reset();pair_head_unavailable=true;
                    OutputDebugStringA((std::string("RIFE optional startup Head unavailable; using original engine: ")+error.what()+"\n").c_str());
                }
            }
        }
        if(!enqueues)return startup_ready()?0:-1;
        startup_warm_value=++external.serial;
        cudaExternalSemaphoreSignalParams signal{};signal.params.fence.value=startup_warm_value;
        cu.check(cu.cudaSignalExternalSemaphoresAsync(&external.semaphore,&signal,1,stream),"Signal startup warmup completion");
        ck(external.fence->SetEventOnCompletion(startup_warm_value,external.ready_event),"Arm startup warmup event");
        startup_warm_pending=true;
        return enqueues;
    }
    bool startup_ready()const noexcept{
        return startup_warmed&&(!startup_head_required||startup_head_warmed||pair_head_unavailable)&&
            (!startup_warm_pending||external.fence->GetCompletedValue()>=startup_warm_value);
    }
    void finish_startup_warmup(){
        if(!startup_warm_pending)return;
        // The engine waits outside the shared D3D lock before entering here.
        if(external.fence->GetCompletedValue()<startup_warm_value)
            throw std::runtime_error("Startup warmup was not completed before guide submission");
        ck(external.context->Wait(external.fence.Get(),startup_warm_value),"Order completed startup warmup");
        ResetEvent(external.ready_event);startup_warm_pending=false;
    }
    void parameters(float phase,unsigned plane=0){p.phase=phase;p.plane=plane;c->UpdateSubresource(params.Get(),0,nullptr,&p,0,0);auto b=params.Get();c->CSSetConstantBuffers(0,1,&b);c->PSSetConstantBuffers(0,1,&b);}
    void unbind(){ID3D11ShaderResourceView* v[5]{};ID3D11UnorderedAccessView* u=nullptr;c->CSSetShaderResources(0,4,v);c->PSSetShaderResources(0,5,v);c->CSSetUnorderedAccessViews(0,1,&u,nullptr);c->CSSetShader(nullptr,nullptr,0);c->OMSetRenderTargets(0,nullptr,nullptr);}
    void clear_history()noexcept{prepared=false;for(auto& v:views)v.Reset();}
    void reset(bool reuse=false)noexcept{
        clear_history();startup_warmed=false;startup_warm_pending=false;startup_warm_value=0;startup_head_required=startup_head_warmed=false;
        if(shared){auto& cu=*shared->cuda;cu.cudaSetDevice(gpu);bool drained=!stream||cu.cudaStreamSynchronize(stream)==cudaSuccess;
            for(auto& r:resources)if(r){cu.cudaGraphicsUnregisterResource(r);r=nullptr;}
            graph.clear(cu);graph.disabled=false;
            if(pair_head){pair_head->recycle=drained;pair_head.reset();}
            external.release(cu);
            profiler.release_cuda();
            if(reuse&&drained)shared->recycle_context(execution);
            execution.reset();if(stream){cu.cudaStreamDestroy(stream);stream=nullptr;}}
        packed_motion.reset();
        for(auto& v:batchSrvs)v.Reset();
        inputUav.Reset();motionSrv.Reset();input.Reset();motion.Reset();params.Reset();prepCS.Reset();timeCS.Reset();vertex.Reset();pixel.Reset();pixelY.Reset();pixelUV.Reset();sampler.Reset();raster.Reset();
    }
    ~Impl(){reset(true);}
    void prepare(ID3D11Texture2D* a,ID3D11Texture2D* b,int w,int h,const GpuWait& wait,const PairReady& ready,const SourcePlanes& planes,const CpuWork& initialization){
        clear_history();
        if(!a||!b||w<2||h<2||(w&1)||(h&1))throw std::invalid_argument("RIFE needs matching even NV12/P010 sources");
        D3D11_TEXTURE2D_DESC ad{},bd{};a->GetDesc(&ad);b->GetDesc(&bd);
        if(ad.Format!=bd.Format||(ad.Format!=DXGI_FORMAT_NV12&&ad.Format!=DXGI_FORMAT_P010)||ad.Width<UINT(w)||ad.Height<UINT(h)||ad.Width!=bd.Width||ad.Height!=bd.Height||ad.ArraySize!=1||bd.ArraySize!=1)throw std::invalid_argument("RIFE source layout mismatch");
        // Finish the scene/identity gate before loading the neural runtime.
        if(wait)wait();if(ready&&!ready())return;
        auto profileStart=profiler.data.enabled?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
        configure_resolution(w,h);initialize(initialization);finish_startup_warmup();
        if(profiler.data.enabled)profiler.data.initialize_cpu_ms=RifeProfiler::ms(profileStart);
        profileStart=profiler.data.enabled?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
        p.w=float(w);p.h=float(h);p.tenBit=ad.Format==DXGI_FORMAT_P010;p.portrait=h>w;
        const float rw=float(p.portrait?h:w),rh=float(p.portrait?w:h);
        const float scale=(std::min)({float(GW)/rw,float(GH)/rh,1.0f});
        p.aw=rw*scale;p.ah=rh*scale;p.ox=(GW-p.aw)*.5f;p.oy=(GH-p.ah)*.5f;
        const bool supplied=std::any_of(planes.begin(),planes.end(),[](auto* view){return view!=nullptr;});
        if(supplied&&!std::all_of(planes.begin(),planes.end(),[](auto* view){return view!=nullptr;}))
            throw std::invalid_argument("RIFE source planes must be complete");
        if(supplied){for(size_t i=0;i<views.size();++i)views[i]=planes[i];}
        else {
            ID3D11Texture2D* textures[]={a,b,a,b};
            for(unsigned i=0;i<4;++i){D3D11_SHADER_RESOURCE_VIEW_DESC sd{};sd.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D;sd.Texture2D.MipLevels=1;
                sd.Format=p.tenBit?(i<2?DXGI_FORMAT_R16_UNORM:DXGI_FORMAT_R16G16_UNORM):(i<2?DXGI_FORMAT_R8_UNORM:DXGI_FORMAT_R8G8_UNORM);
                ck(d->CreateShaderResourceView(textures[i],&sd,&views[i]),"RIFE source plane");}
        }
        auto guideTimer=profiler.begin_gpu(RifeGpuKind::guide,.5f,w,h);
        parameters(.5f);ID3D11ShaderResourceView* srvs[]={views[0].Get(),views[1].Get(),views[2].Get(),views[3].Get()};
        c->CSSetShaderResources(0,4,srvs);auto sp=sampler.Get();c->CSSetSamplers(0,1,&sp);auto u=inputUav.Get();c->CSSetUnorderedAccessViews(0,1,&u,nullptr);
        c->CSSetShader(prepCS.Get(),nullptr,0);c->Dispatch((GW+15)/16,(GH+15)/16,1);unbind();profiler.end_gpu(guideTimer);input_phase=.5f;prepared=true;
        if(profiler.data.enabled)profiler.data.guide_submit_cpu_ms=RifeProfiler::ms(profileStart);
    }
    void acquire_buffers(){auto& cu=*shared->cuda;if(external.active)external.acquire(cu,stream,startup_cuda_work?*startup_cuda_work:CpuWork{});else cu.check(cu.cudaGraphicsMapResources(2,resources.data(),stream),"Map RIFE D3D11 buffers");++handoffs;}
    void release_buffers(){auto& cu=*shared->cuda;if(external.active)external.relinquish(cu,stream,completion_wait?*completion_wait:CompletionWait{},startup_cuda_work?*startup_cuda_work:CpuWork{});else cu.check(cu.cudaGraphicsUnmapResources(2,resources.data(),stream),"Unmap RIFE D3D11 buffers");}
    void buffer_pointers(std::array<void*,2>& pointers,size_t* bytes){
        auto& cu=*shared->cuda;
        for(unsigned i=0;i<2;++i)if(external.active){pointers[i]=external.pointers[i];bytes[i]=external.sizes[i];}else cu.check(cu.cudaGraphicsResourceGetMappedPointer(&pointers[i],&bytes[i],resources[i]),"Get RIFE CUDA buffer");
    }
    void infer(float phase,RifePhaseTiming* sample){
        auto tick=sample?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
        auto mark=[&](double RifePhaseTiming::*field){if(sample){sample->*field=RifeProfiler::ms(tick);tick=RifeProfiler::Clock::now();}};
        // prepareInput already fills t=.5. Do not dispatch a second write for
        // midpoint-only pairs, or repeated requests for an unchanged phase.
        if(reference_interop||phase!=input_phase){
            auto timer=profiler.begin_gpu(RifeGpuKind::time,phase,unsigned(p.w),unsigned(p.h));
            parameters(phase);auto u=inputUav.Get();c->CSSetUnorderedAccessViews(0,1,&u,nullptr);c->CSSetShader(timeCS.Get(),nullptr,0);c->Dispatch((GW+15)/16,(GH+15)/16,1);unbind();profiler.end_gpu(timer);
            input_phase=phase;
        }
        // MapResources orders preceding graphics work; separate Flush duplicates
        // submission. The reference switch exists only for controlled A/B tests.
        if(reference_interop)c->Flush();
        mark(&RifePhaseTiming::time_submit_cpu_ms);
        auto& cu=*shared->cuda;cu.check(cu.cudaSetDevice(gpu),"Select RIFE inference GPU");
        mark(&RifePhaseTiming::device_cpu_ms);profiler.record(sample,0);
        if(sample)tick=RifeProfiler::Clock::now();
        acquire_buffers();
        if(sample){sample->external_fence=external.active;sample->map_count=external.active?0:1;sample->unmap_count=external.active?0:1;}
        mark(&RifePhaseTiming::map_cpu_ms);profiler.record(sample,1);
        if(sample)tick=RifeProfiler::Clock::now();
        bool mapped=true;
        try{
            std::array<void*,2> pointers{};size_t bytes[2]{};
            buffer_pointers(pointers,bytes);
            if(bytes[0]<N*7*4||bytes[1]<N*5*4)throw std::runtime_error("RIFE CUDA buffer size mismatch");
            mark(&RifePhaseTiming::bind_cpu_ms);profiler.record(sample,2);
            if(sample)tick=RifeProfiler::Clock::now();
            graph.run(cu,*execution,stream,pointers,graph_enabled);
            if(sample){sample->graph_replayed=graph.replayed;sample->graph_created=graph.built;}
            mark(&RifePhaseTiming::enqueue_cpu_ms);profiler.record(sample,3);profiler.record(sample,4);
            if(sample)tick=RifeProfiler::Clock::now();
            release_buffers();mapped=false;
            mark(&RifePhaseTiming::unmap_cpu_ms);profiler.record(sample,5);
            // Unmap orders prior CUDA work before subsequently submitted D3D11
            // work. Preserve that GPU dependency without blocking the CPU here.
            // D3D surface consumers already use context ordering/completion.
            if(force_cuda_sync){
                if(sample)tick=RifeProfiler::Clock::now();
                cu.check(cu.cudaStreamSynchronize(stream),"Complete RIFE inference (reference mode)");
                mark(&RifePhaseTiming::sync_cpu_ms);
            }
            if(sample)tick=RifeProfiler::Clock::now();
            auto status=cu.cudaStreamQuery(stream);
            if(status!=cudaErrorNotReady)cu.check(status,"RIFE asynchronous execution status");
            mark(&RifePhaseTiming::status_query_cpu_ms);++inferences;
        }catch(...){if(mapped)try{release_buffers();}catch(...){}cu.cudaStreamSynchronize(stream);throw;}
    }
    void composite(ID3D11RenderTargetView* y,ID3D11RenderTargetView* uv,float phase,ID3D11ShaderResourceView* motionView,RifePhaseTiming* sample){
        if(reference_composite)parameters(phase);
        auto compositeStart=sample?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
        auto compositeTimer=profiler.begin_gpu(RifeGpuKind::composite,phase,unsigned(p.w),unsigned(p.h));
        if(packed_motion)packed_motion->pack(motionView,params.Get());
        if(sample)sample->packed_layout=bool(packed_motion);
        c->IASetInputLayout(nullptr);c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);c->VSSetShader(vertex.Get(),nullptr,0);c->PSSetShader(pixel.Get(),nullptr,0);
        c->RSSetState(raster.Get());c->OMSetBlendState(nullptr,nullptr,0xffffffff);c->OMSetDepthStencilState(nullptr,0);
        ID3D11ShaderResourceView* v[]={views[0].Get(),views[1].Get(),views[2].Get(),views[3].Get(),motionView};c->PSSetShaderResources(0,5,v);auto sp=sampler.Get();c->PSSetSamplers(0,1,&sp);
        // Specialized shaders do not read the phase or plane fields. Both draws
        // reuse the last parameter upload; avoid two per-output buffer updates.
        auto cb=params.Get();c->PSSetConstantBuffers(0,1,&cb);
        for(unsigned plane=0;plane<2;++plane){if(reference_composite)parameters(phase,plane);c->PSSetShader(reference_composite?pixel.Get():(packed_motion?(plane?packed_motion->uv_shader.Get():packed_motion->y_shader.Get()):(plane?pixelUV.Get():pixelY.Get())),nullptr,0);auto rt=plane?uv:y;D3D11_VIEWPORT vp{0,0,p.w/(plane?2:1),p.h/(plane?2:1),0,1};c->RSSetViewports(1,&vp);c->OMSetRenderTargets(1,&rt,nullptr);c->Draw(3,0);}
        if(packed_motion)packed_motion->unbind();
        unbind();profiler.end_gpu(compositeTimer);
        if(sample){sample->composite_submit_cpu_ms=RifeProfiler::ms(compositeStart);sample->total_cpu_ms+=sample->composite_submit_cpu_ms;
            if(startup_cuda_work)(*startup_cuda_work)([&]{profiler.finish(sample);});else profiler.finish(sample);}
    }
    void render(ID3D11RenderTargetView* y,ID3D11RenderTargetView* uv,float phase){
        if(!prepared||!y||!uv||!std::isfinite(phase)||phase<=0||phase>=1)throw std::invalid_argument("RIFE phase is not prepared");
        auto started=profiler.data.enabled?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
        auto sample=profiler.phase(phase,unsigned(p.w),unsigned(p.h));
        infer(phase,sample);if(sample)sample->total_cpu_ms=RifeProfiler::ms(started);
        composite(y,uv,phase,motionSrv.Get(),sample);
    }
    void render_batch(const PhaseTarget* targets,size_t count,const CompletionWait& wait,const CpuWork& optional_setup,const std::atomic<bool>* cancelled){
        const auto checkpoint=[&]{if(cancelled&&cancelled->load(std::memory_order_acquire))throw GpuWorkCancelled{};};
        checkpoint();
        struct Scope {Impl& impl;~Scope(){impl.completion_wait=nullptr;impl.startup_cuda_work=nullptr;}} scope{*this};
        completion_wait=environment_flag(L"RIFE_REFERENCE_QUEUED_WAIT")?nullptr:&wait;
        if(!prepared||!targets||!count||count>32)throw std::invalid_argument("Invalid RIFE phase batch");
        for(size_t i=0;i<count;++i)if(!targets[i].y||!targets[i].uv||!std::isfinite(targets[i].fraction)||targets[i].fraction<=0||targets[i].fraction>=1)
            throw std::invalid_argument("Invalid RIFE batch phase");
        if(!batch_enabled||count==1){for(size_t i=0;i<count;++i)render(targets[i].y,targets[i].uv,targets[i].fraction);return;}
        // First inference and first-use E03 preparation opt in. Stable playback
        // and single-output/legacy interop retain their original scopes.
        startup_cuda_work=optional_setup&&external.active?&optional_setup:nullptr;
        auto cuda_only=[&](auto&& work){if(startup_cuda_work)(*startup_cuda_work)(work);else work();};
        const auto headTick=profiler.data.enabled?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
        auto& cu=*shared->cuda;
        auto prepare_optional=[&]{
        cu.check(cu.cudaSetDevice(gpu),"Select batched RIFE GPU");
        // Exactly one output keeps the original path above, without loading or
        // allocating split engines. Multi-output reuse is local to this call;
        // even identical textures/PTS after a seek always recompute the Head.
        if(pair_head_requested&&!pair_head_unavailable&&!pair_head){
            try{
                std::shared_ptr<RifeHeadPlans> plans;
                if(environment_flag(L"RIFE_HEAD_CACHE_SYNC_PREPARE"))plans=cached_head_plans(shared,directory); // Deterministic correctness probes only.
                else {
                    if(!head_preparation)head_preparation=request_head_preparation(shared,directory);
                    std::lock_guard<std::mutex> lock(head_preparation->mutex);
                    plans=head_preparation->ready;pair_head_unavailable=head_preparation->failed;
                }
                if(plans){auto next=std::make_unique<RifePairHeadCache>(std::move(plans));next->initialize();pair_head=std::move(next);}
            }
            catch(const std::exception& error){pair_head_unavailable=true;OutputDebugStringA((std::string("RIFE optional pair Head unavailable; using original engine: ")+error.what()+"\n").c_str());}
            if(profiler.data.enabled)profiler.data.head_prepare_cpu_ms=RifeProfiler::ms(headTick);
        }
        };
        // Module loading on another thread can serialize cudaSetDevice even
        // when the playback thread is already on the same GPU. This block has
        // no immediate-context commands or source-image work.
        cuda_only(prepare_optional);
        checkpoint();
        bool pair_head_submitted=false;
        // Bound motion storage to four guide-sized results (~40 MiB), regardless
        // of source resolution or requested output multiplier. Each chunk owns
        // both shared buffers until every CUDA result is preserved. No D3D draw
        // accesses them while mapped. The same TRT bindings/graph serve all phases.
        for(size_t base=0;base<count;base+=PhaseCapacity){
            const auto size=(std::min)(size_t(PhaseCapacity),count-base);
            std::array<RifePhaseTiming*,PhaseCapacity> samples{};
            cuda_only([&]{for(size_t i=0;i<size;++i){samples[i]=profiler.phase(targets[base+i].fraction,unsigned(p.w),unsigned(p.h));if(samples[i]){samples[i]->batch_size=unsigned(size);samples[i]->external_fence=external.active;}}});
            auto start=profiler.data.enabled?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
            cuda_only([&]{profiler.record(samples[0],0);});
            acquire_buffers();
            if(samples[0]){samples[0]->map_cpu_ms=RifeProfiler::ms(start);samples[0]->map_count=external.active?0:1;}
            bool mapped=true,submitted=false;
            try{
                cuda_only([&]{
                profiler.record(samples[0],1);
                std::array<void*,2> pointers{};size_t bytes[2]{};
                buffer_pointers(pointers,bytes);
                constexpr size_t motionBytes=size_t(N)*5*sizeof(float);
                if(bytes[0]<size_t(N)*7*sizeof(float)||bytes[1]<motionBytes*(PhaseCapacity+1))throw std::runtime_error("RIFE batch buffer size mismatch");
                for(size_t i=0;i<size;++i){
                    checkpoint();
                    auto sample=samples[i];auto phase=targets[base+i].fraction;
                    auto tick=sample?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
                    if(i){profiler.record(sample,0);profiler.record(sample,1);}
                    // Exact float bit pattern, asynchronous on the inference stream.
                    // Only the guide's time plane is modified, never source surfaces.
                    submitted=true;
                    cu.fill_phase(static_cast<float*>(pointers[0])+N*6,phase,stream);input_phase=phase;
                    if(sample)sample->time_submit_cpu_ms=RifeProfiler::ms(tick);
                    profiler.record(sample,2);tick=sample?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
                    if(pair_head){
                        if(!pair_head_submitted){pair_head->begin_pair(cu,stream,pointers[0],graph_enabled);pair_head_submitted=true;if(sample)sample->head_cache_first=true;}
                        pair_head->phase(cu,stream,pointers,graph_enabled);
                    }else graph.run(cu,*execution,stream,pointers,graph_enabled);
                    checkpoint();
                    const auto& active_graph=pair_head?pair_head->tail_graph:graph;
                    if(sample){sample->head_cached=bool(pair_head);sample->enqueue_cpu_ms=RifeProfiler::ms(tick);sample->graph_replayed=active_graph.replayed;sample->graph_created=active_graph.built;}
                    profiler.record(sample,3);tick=sample?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
                    // Earlier phases must survive the next fixed-address TRT write.
                    // The final phase remains in scratch until this chunk's draws:
                    // the next acquire still fences those reads before CUDA reuse.
                    if(!elide_final_batch_copy || i+1<size) {
                        cu.check(cu.cudaMemcpyAsync(static_cast<char*>(pointers[1])+motionBytes*(i+1),pointers[1],motionBytes,cudaMemcpyDeviceToDevice,stream),"Preserve batched RIFE motion");
                        if(sample){sample->copy_count=1;sample->copy_cpu_ms=RifeProfiler::ms(tick);}
                    }
                    profiler.record(sample,4);++inferences;
                    if(i+1<size)profiler.record(sample,5);
                    if(sample)sample->total_cpu_ms=sample->map_cpu_ms+sample->time_submit_cpu_ms+sample->enqueue_cpu_ms+sample->copy_cpu_ms;
                }
                });
                auto last=samples[size-1];start=last?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};
                release_buffers();mapped=false;
                if(last){last->unmap_count=external.active?0:1;last->unmap_cpu_ms=RifeProfiler::ms(start);last->total_cpu_ms+=last->unmap_cpu_ms;}
                cuda_only([&]{
                profiler.record(last,5);
                if(force_cuda_sync){start=last?RifeProfiler::Clock::now():RifeProfiler::Clock::time_point{};cu.check(cu.cudaStreamSynchronize(stream),"Complete batched RIFE (reference mode)");if(last)last->sync_cpu_ms=RifeProfiler::ms(start);}
                auto status=cu.cudaStreamQuery(stream);if(status!=cudaErrorNotReady)cu.check(status,"Batched RIFE asynchronous execution status");
                });
            }catch(const GpuWorkCancelled&){
                // A satisfied external fence already proves every submitted
                // CUDA operation complete. Do not enter another CUDA API while
                // the optional module loader may hold its driver lock.
                bool completed=false;
                // A queued CUDA wait alone never accesses these buffers. With
                // no subsequent writes there is nothing to hand back to D3D;
                // its guide commands remain ordered on its own context.
                if(external.active&&!submitted)completed=true;
                else if(mapped){release_buffers();completed=external.active&&completion_wait&&bool(*completion_wait);}
                if(!completed)cu.check(cu.cudaStreamSynchronize(stream),"Drain cancelled RIFE batch");
                profiler.discard();throw;
            }catch(...){if(mapped)try{release_buffers();}catch(...){}cu.cudaStreamSynchronize(stream);throw;}
            // Unmap orders every phase before these unchanged native-resolution draws.
            for(size_t i=0;i<size;++i)composite(targets[base+i].y,targets[base+i].uv,targets[base+i].fraction,
                elide_final_batch_copy&&i+1==size?motionSrv.Get():batchSrvs[i].Get(),samples[i]);
        }
    }
};
MotionSynthesizer::MotionSynthesizer(ID3D11Device* d,ID3D11DeviceContext* c,const std::filesystem::path& directory):impl_(std::make_unique<Impl>(d,c,directory)){}
MotionSynthesizer::~MotionSynthesizer()=default;
void MotionSynthesizer::prepare(ID3D11Texture2D* a,ID3D11Texture2D* b,int w,int h,const GpuWait& wait,const PairReady& ready,const SourcePlanes& planes,const CpuWork& initialization){impl_->prepare(a,b,w,h,wait,ready,planes,initialization);}
void MotionSynthesizer::set_hdr_transfer(unsigned transfer)noexcept{impl_->p.hdrTransfer=transfer;}
void MotionSynthesizer::begin_timing(bool enabled,int64_t pts){impl_->profiler.begin_pair(enabled,pts);}
const RifeTiming& MotionSynthesizer::timing()const noexcept{return impl_->profiler.data;}
bool MotionSynthesizer::prepared()const noexcept{return impl_->prepared;}
int MotionSynthesizer::startup_warmup(int width,int height,bool multiple){return impl_->startup_warmup(width,height,multiple);}
bool MotionSynthesizer::startup_ready()const noexcept{return impl_->startup_ready();}
HANDLE MotionSynthesizer::startup_warmup_event()const noexcept{return impl_->startup_warm_pending && impl_->external.fence->GetCompletedValue()<impl_->startup_warm_value?impl_->external.ready_event:nullptr;}
MotionAnalysisInfo MotionSynthesizer::analysis_info()const noexcept{return {GW,GH};}
void MotionSynthesizer::render_phase(ID3D11RenderTargetView* y,ID3D11RenderTargetView* uv,float t){impl_->render(y,uv,t);}
bool MotionSynthesizer::initialization_pending()const noexcept{return !impl_->execution;}
bool MotionSynthesizer::optional_setup_pending(size_t count)const noexcept{return count>1&&impl_->batch_enabled&&(!impl_->graph.warmed||(impl_->pair_head_requested&&!impl_->pair_head_unavailable&&!impl_->pair_head));}
void MotionSynthesizer::render_phases(const PhaseTarget* targets,size_t count,const CompletionWait& wait,const CpuWork& optional_setup,const std::atomic<bool>* cancelled){impl_->render_batch(targets,count,wait,optional_setup,cancelled);}
uint64_t MotionSynthesizer::handoff_count()noexcept{return handoffs.load();}
uint64_t MotionSynthesizer::head_cache_pair_count()noexcept{return head_pairs.load();}
uint64_t MotionSynthesizer::head_cache_phase_count()noexcept{return head_phases.load();}
uint64_t MotionSynthesizer::head_cache_load_count()noexcept{return head_plan_loads.load();}
uint64_t MotionSynthesizer::head_cache_context_count()noexcept{return head_context_creates.load();}
void MotionSynthesizer::invalidate_history()noexcept{impl_->clear_history();}
void MotionSynthesizer::reset()noexcept{impl_->reset();}
bool MotionSynthesizer::preload_async(const std::filesystem::path& dir,ID3D11Device* device)noexcept{try{return request_preload(dir.empty()?default_runtime():dir,device);}catch(...){return false;}}
bool MotionSynthesizer::preload_adapter_async(const std::filesystem::path& dir,unsigned index)noexcept{
    try{ComPtr<IDXGIFactory1> factory;ComPtr<IDXGIAdapter1> adapter;
        ck(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"Preload adapter factory");ck(factory->EnumAdapters1(index,&adapter),"Preload selected adapter");
        return request_adapter_preload(dir.empty()?default_runtime():dir,adapter.Get());
    }catch(...){return false;}
}
uint64_t MotionSynthesizer::preload_pending()noexcept{return preload_jobs.load();}
uint64_t MotionSynthesizer::optional_preload_pending()noexcept{return head_preparation_jobs.load();}
uint64_t MotionSynthesizer::execution_create_count()noexcept{return contexts_created.load();}
uint64_t MotionSynthesizer::engine_load_count()noexcept{return loads.load();}
uint64_t MotionSynthesizer::inference_count()noexcept{return inferences.load();}
uint64_t MotionSynthesizer::graph_launch_count()noexcept{return graph_launches.load();}
uint64_t MotionSynthesizer::graph_build_count()noexcept{return graph_builds.load();}
uint64_t MotionSynthesizer::graph_fallback_count()noexcept{return graph_fallbacks.load();}
}
