// Legacy D3D11 shared buffer handles are non-owning. Keep the D3D resources
// alive until CUDA mappings/imports are destroyed. A shared D3D11 fence orders
// the same graphics/CUDA accesses as map/unmap, without global interop barriers.
struct RifeExternalBuffers {
    ComPtr<ID3D11DeviceContext4> context;
    ComPtr<ID3D11Fence> fence;
    std::array<cudaExternalMemory_t,2> memory{};
    std::array<void*,2> pointers{};
    std::array<size_t,2> sizes{};
    cudaExternalSemaphore_t semaphore=nullptr;
    uint64_t serial=0;
    bool active=false;
    HANDLE ready_event=nullptr;
    void initialize(Cuda& cu,ID3D11Device* device,ID3D11DeviceContext* c,ID3D11Buffer* input,ID3D11Buffer* motion){
        try{
            ComPtr<ID3D11Device5>d5;ck(device->QueryInterface(IID_PPV_ARGS(&d5)),"RIFE fence device");ck(c->QueryInterface(IID_PPV_ARGS(&context)),"RIFE fence context");
            ck(d5->CreateFence(0,D3D11_FENCE_FLAG_SHARED,IID_PPV_ARGS(&fence)),"RIFE interop fence");
            HANDLE handle=nullptr;ck(fence->CreateSharedHandle(nullptr,GENERIC_ALL,nullptr,&handle),"RIFE interop fence handle");
            cudaExternalSemaphoreHandleDesc sd{};sd.type=cudaExternalSemaphoreHandleTypeD3D11Fence;sd.handle.win32.handle=handle;
            const auto imported=cu.cudaImportExternalSemaphore(&semaphore,&sd);CloseHandle(handle);cu.check(imported,"Import RIFE interop fence");
            ID3D11Buffer* buffers[]={input,motion};
            for(unsigned i=0;i<2;++i){
                D3D11_BUFFER_DESC desc{};buffers[i]->GetDesc(&desc);sizes[i]=desc.ByteWidth;
                ComPtr<IDXGIResource> resource;ck(buffers[i]->QueryInterface(IID_PPV_ARGS(&resource)),"RIFE shared buffer interface");
                ck(resource->GetSharedHandle(&handle),"RIFE shared buffer handle"); // KMT handle: never CloseHandle.
                cudaExternalMemoryHandleDesc md{};md.type=cudaExternalMemoryHandleTypeD3D11ResourceKmt;md.handle.win32.handle=handle;md.size=sizes[i];md.flags=cudaExternalMemoryDedicated;
                cu.check(cu.cudaImportExternalMemory(&memory[i],&md),"Import RIFE shared buffer");
                cudaExternalMemoryBufferDesc map{};map.size=sizes[i];cu.check(cu.cudaExternalMemoryGetMappedBuffer(&pointers[i],memory[i],&map),"Map persistent RIFE buffer");
            }
            if(environment_flag(L"RIFE_TEST_EXTERNAL_IMPORT_FAILURE"))throw std::runtime_error("Injected external import failure");
            ready_event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!ready_event)throw std::runtime_error("Cannot create RIFE ready event");
            active=true;
        }catch(...){release(cu);throw;}
    }
    void acquire(Cuda& cu,cudaStream_t stream,const MotionSynthesizer::CpuWork& startup_work={}){
        const auto value=++serial;
        ck(context->Signal(fence.Get(),value),"Signal RIFE guide readiness");context->Flush();
        cudaExternalSemaphoreWaitParams wait{};wait.params.fence.value=value;
        auto enqueue=[&]{cu.check(cu.cudaWaitExternalSemaphoresAsync(&semaphore,&wait,1,stream),"Wait for RIFE guide on CUDA");};
        if(startup_work)startup_work(enqueue);else enqueue();
    }
    void relinquish(Cuda& cu,cudaStream_t stream,const MotionSynthesizer::CompletionWait& completion_wait,const MotionSynthesizer::CpuWork& startup_work={}){
        const auto value=++serial;
        cudaExternalSemaphoreSignalParams signal{};signal.params.fence.value=value;
        auto enqueue=[&]{cu.check(cu.cudaSignalExternalSemaphoresAsync(&semaphore,&signal,1,stream),"Signal RIFE motion readiness");};
        if(startup_work)startup_work(enqueue);else enqueue();
        if(completion_wait&&fence->GetCompletedValue()<value){
            ck(fence->SetEventOnCompletion(value,ready_event),"Arm RIFE CUDA ready event");
            completion_wait(ready_event);
        }
        // Keep the formal cross-API memory dependency, but when a wait service
        // exists its fence is ALREADY satisfied: do not park the shared queue.
        ck(context->Wait(fence.Get(),value),"Wait for RIFE motion on D3D11");
    }
    void release(Cuda& cu)noexcept{
        // Called after the inference stream is drained. Complete the final
        // graphics reads before freeing persistent mappings. This is teardown only.
        if(active&&context&&fence){
            const auto value=++serial;
            if(SUCCEEDED(context->Signal(fence.Get(),value))){context->Flush();
                HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
                if(event){if(SUCCEEDED(fence->SetEventOnCompletion(value,event)))WaitForSingleObject(event,10000);CloseHandle(event);}
            }
        }
        active=false;
        if(ready_event){CloseHandle(ready_event);ready_event=nullptr;}
        for(unsigned i=0;i<2;++i){if(pointers[i]){cu.cudaFree(pointers[i]);pointers[i]=nullptr;}if(memory[i]){cu.cudaDestroyExternalMemory(memory[i]);memory[i]=nullptr;}}
        if(semaphore){cu.cudaDestroyExternalSemaphore(semaphore);semaphore=nullptr;}
        fence.Reset();context.Reset();sizes={};serial=0;
    }
};
