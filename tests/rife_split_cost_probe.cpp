// Offline pair-local Head cache cost screen. No decoder, D3D composition or player.
#define wmain unused_inspection_main
#include "../tools/inspect_rife_engine.cpp"
#undef wmain
#include <cuda.h>
#include <chrono>
#include <cstring>
#include <cstdint>
struct CostPlan {
    std::unique_ptr<ICudaEngine> engine;std::unique_ptr<IExecutionContext> context;
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    CostPlan(IRuntime& runtime,const fs::path& path){
        std::ifstream file(path,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});need(!bytes.empty(),"engine file");
        engine.reset(runtime.deserializeCudaEngine(bytes.data(),bytes.size()));need(bool(engine),"engine");context.reset(engine->createExecutionContext());need(bool(context),"context");
        Dims dims{};dims.nbDims=4;dims.d[0]=1;dims.d[1]=7;dims.d[2]=540;dims.d[3]=960;need(context->setInputShape("input",dims),"input shape");
    }
    void bind(const char* name,void* p){need(context->setTensorAddress(name,p),"bind");}
};
int wmain(int argc,wchar_t** argv){try {
    need(argc==6,"runtime original.engine head.engine tail.engine results.csv");fs::path dir=fs::absolute(argv[1]);
    Lib infer(dir/L"nvinfer_10.dll"),plugins(dir/L"nvinfer_plugin_10.dll"),cuda(dir/L"cudart64_13.dll");Log logger;
    using Init=bool(__cdecl*)(void*,const char*);need(plugins.get<Init>("initLibNvInferPlugins")(&logger,""),"plugins");
    using Create=void*(__cdecl*)(void*,int);std::unique_ptr<IRuntime> runtime(static_cast<IRuntime*>(infer.get<Create>("createInferRuntime_INTERNAL")(&logger,NV_TENSORRT_VERSION)));need(bool(runtime),"runtime");
    wchar_t system[MAX_PATH]{};need(GetSystemDirectoryW(system,MAX_PATH)>0,"system directory");Lib driver(fs::path(system)/L"nvcuda.dll"),nvml(fs::path(system)/L"nvml.dll");
    auto fill=driver.get<decltype(&::cuMemsetD32Async)>("cuMemsetD32Async");
    auto nvInit=nvml.get<int(*)()>("nvmlInit_v2");auto nvClose=nvml.get<int(*)()>("nvmlShutdown");
    auto nvDevice=nvml.get<int(*)(unsigned,void**)>("nvmlDeviceGetHandleByIndex_v2");auto nvEnergy=nvml.get<int(*)(void*,unsigned long long*)>("nvmlDeviceGetTotalEnergyConsumption");
    need(nvInit()==0,"NVML init");void* device=nullptr;need(nvDevice(0,&device)==0,"NVML device");
#define FN(n) auto n=cuda.get<decltype(&::n)>(#n)
    FN(cudaSetDevice);FN(cudaMalloc);FN(cudaFree);FN(cudaMemcpy);FN(cudaStreamCreate);FN(cudaStreamDestroy);FN(cudaStreamSynchronize);
    FN(cudaStreamBeginCapture);FN(cudaStreamEndCapture);FN(cudaGraphInstantiateWithFlags);FN(cudaGraphLaunch);FN(cudaGraphDestroy);FN(cudaGraphExecDestroy);
    FN(cudaEventCreate);FN(cudaEventRecord);FN(cudaEventSynchronize);FN(cudaEventElapsedTime);FN(cudaEventDestroy);
#undef FN
    need(cudaSetDevice(0)==cudaSuccess,"device");CostPlan baseline(*runtime,argv[2]),head(*runtime,argv[3]),tail(*runtime,argv[4]);
    Dims dims{};dims.nbDims=4;dims.d[0]=2;dims.d[1]=4;dims.d[2]=544;dims.d[3]=960;need(tail.context->setInputShape("features",dims),"feature shape");
    constexpr size_t N=960*540,F=2*4*544*960;void* input=nullptr;void* features=nullptr;void* output=nullptr;
    std::cout<<"MEMORY context_workspace_bytes baseline="<<baseline.engine->getDeviceMemorySizeV2()
             <<" head="<<head.engine->getDeviceMemorySizeV2()<<" tail="<<tail.engine->getDeviceMemorySizeV2()
             <<" features="<<F*4<<" (excludes plan weights, graph allocations and driver overhead)\n";
    need(cudaMalloc(&input,N*7*4)==cudaSuccess&&cudaMalloc(&features,F*4)==cudaSuccess&&cudaMalloc(&output,N*5*4)==cudaSuccess,"buffers");
    std::vector<float> host(N*7);for(size_t i=0;i<host.size();++i)host[i]=float(i%997)/996.f;
    need(cudaMemcpy(input,host.data(),host.size()*4,cudaMemcpyHostToDevice)==cudaSuccess,"input");
    baseline.bind("input",input);baseline.bind("motion",output);head.bind("input",input);head.bind("features",features);tail.bind("input",input);tail.bind("features",features);tail.bind("motion",output);
    cudaStream_t stream=nullptr;need(cudaStreamCreate(&stream)==cudaSuccess,"stream");
    for(auto* p:{&baseline,&head,&tail}){
        need(p->context->enqueueV3(stream),"warmup");need(cudaStreamSynchronize(stream)==cudaSuccess,"warmup done");
        need(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"capture start");need(p->context->enqueueV3(stream),"capture enqueue");
        need(cudaStreamEndCapture(stream,&p->graph)==cudaSuccess,"capture end");need(cudaGraphInstantiateWithFlags(&p->executable,p->graph,0)==cudaSuccess,"instantiate");
    }
    cudaEvent_t start=nullptr,stop=nullptr;need(cudaEventCreate(&start)==cudaSuccess&&cudaEventCreate(&stop)==cudaSuccess,"events");
    auto pair=[&](bool cached,int phases){
        if(cached)need(cudaGraphLaunch(head.executable,stream)==cudaSuccess,"head launch");
        for(int t=1;t<=phases;++t){float phase=float(t)/(phases+1);unsigned bits=0;std::memcpy(&bits,&phase,4);need(fill(reinterpret_cast<CUdeviceptr>(static_cast<float*>(input)+N*6),bits,N,reinterpret_cast<CUstream>(stream))==CUDA_SUCCESS,"fill phase");need(cudaGraphLaunch(cached?tail.executable:baseline.executable,stream)==cudaSuccess,"motion launch");}
    };
    std::ofstream csv(argv[5]);need(bool(csv),"report");csv<<"phases,pair,mode,iterations,gpu_span_ms_per_pair,wall_ms_per_pair,board_j_per_pair,extra_feature_bytes\n";
    constexpr int iterations=256;
    for(int phases:{1,4,8})for(int repetition=0;repetition<8;++repetition)for(int order=0;order<2;++order){
        bool cached=((repetition+order)%2)==1;
        for(int i=0;i<24;++i)pair(cached,phases);need(cudaStreamSynchronize(stream)==cudaSuccess,"warm complete");
        unsigned long long before=0,after=0;need(nvEnergy(device,&before)==0,"energy before");auto wallStart=std::chrono::steady_clock::now();
        need(cudaEventRecord(start,stream)==cudaSuccess,"start");for(int i=0;i<iterations;++i)pair(cached,phases);need(cudaEventRecord(stop,stream)==cudaSuccess&&cudaEventSynchronize(stop)==cudaSuccess,"stop");
        const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wallStart).count();need(nvEnergy(device,&after)==0,"energy after");
        float ms=0;need(cudaEventElapsedTime(&ms,start,stop)==cudaSuccess,"elapsed");csv<<phases<<','<<repetition<<','<<(cached?"pair-cache":"reference")<<','<<iterations<<','<<ms/iterations<<','<<wall/iterations<<','<<double(after-before)/1000/iterations<<','<<(cached?F*4:0)<<'\n';csv.flush();
        std::cout<<"phases="<<phases<<" pair="<<repetition<<" cached="<<cached<<" ms="<<ms/iterations<<std::endl;
    }
    for(auto* p:{&baseline,&head,&tail}){cudaGraphExecDestroy(p->executable);cudaGraphDestroy(p->graph);}cudaEventDestroy(stop);cudaEventDestroy(start);cudaFree(output);cudaFree(features);cudaFree(input);cudaStreamDestroy(stream);nvClose();return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
