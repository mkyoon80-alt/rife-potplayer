// Fixed-shape TensorRT work only. Graphics ownership transitions stay outside
// capture. A mapped address is checked on every launch; stale pointers cannot
// replay a graph. A failed capture disables this optimization for the context.
struct RifeGraph {
    cudaGraphExec_t executable=nullptr;
    std::array<void*,2> bindings{};
    bool warmed=false,disabled=false,replayed=false,built=false;
    void clear(Cuda& cu) noexcept {
        if(executable){cu.cudaGraphExecDestroy(executable);executable=nullptr;}
        bindings={};warmed=false;
    }
    template<class Context> void run(Cuda& cu,Context& context,cudaStream_t stream,const std::array<void*,2>& pointers,bool enabled){
        replayed=built=false;
        if(bindings!=pointers){
            // Only a driver remap changes these addresses. Keep context edits
            // out of flight even when the ordinary path is asynchronous.
            if(warmed)cu.check(cu.cudaStreamSynchronize(stream),"RIFE changed-address boundary");
            clear(cu);bindings=pointers;
            if(!context.setTensorAddress("input",bindings[0])||!context.setTensorAddress("motion",bindings[1]))throw std::runtime_error("RIFE tensor binding failed");
        }
        if(executable){cu.check(cu.cudaGraphLaunch(executable,stream),"Replay RIFE CUDA graph");replayed=true;++graph_launches;return;}
        if(enabled&&!disabled&&warmed){
            cudaGraph_t captured=nullptr;auto begin=cu.cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal);
            if(begin==cudaSuccess){
                const bool enqueued=context.enqueueV3(stream);
                // EndCapture is mandatory even if enqueue failed.
                const auto end=cu.cudaStreamEndCapture(stream,&captured);
                if(enqueued&&end==cudaSuccess&&captured){
                    auto status=cu.cudaGraphInstantiateWithFlags(&executable,captured,0);
                    cu.cudaGraphDestroy(captured);captured=nullptr;
                    if(status==cudaSuccess){
                        ++graph_builds;built=true;
                        cu.check(cu.cudaGraphLaunch(executable,stream),"Launch new RIFE CUDA graph");replayed=true;++graph_launches;return;
                    }
                    if(executable){cu.cudaGraphExecDestroy(executable);executable=nullptr;}
                }
                if(captured)cu.cudaGraphDestroy(captured);
            }
            // The standard enqueue remains valid after an unsupported capture.
            // Never retry capture for every frame on an incompatible engine.
            disabled=true;++graph_fallbacks;
        }
        if(!context.enqueueV3(stream))throw std::runtime_error("RIFE inference enqueue failed");
        warmed=true;
    }
};
