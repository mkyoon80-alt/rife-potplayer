// Offline E03 numerical preflight. This does not implement a player cache.
#define wmain unused_inspection_main
#include "../tools/inspect_rife_engine.cpp"
#undef wmain
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
struct Plan {
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    Plan(IRuntime& runtime,const fs::path& path){
        std::ifstream file(path,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
        need(!bytes.empty(),"plan bytes");engine.reset(runtime.deserializeCudaEngine(bytes.data(),bytes.size()));need(bool(engine),"plan deserialize");
        context.reset(engine->createExecutionContext());need(bool(context),"plan context");
        Dims dims{};dims.nbDims=4;dims.d[0]=1;dims.d[1]=7;dims.d[2]=540;dims.d[3]=960;
        need(context->setInputShape("input",dims),"plan input shape");
    }
    void bind(const char* name,void* p){need(context->setTensorAddress(name,p),"plan address");}
    void run(cudaStream_t stream){need(context->enqueueV3(stream),"plan enqueue");}
};
int wmain(int argc,wchar_t** argv){try {
    need(argc==6,"runtime baseline.engine head.engine tail.engine report.json");fs::path dir=fs::absolute(argv[1]);
    Lib infer(dir/L"nvinfer_10.dll"),plugins(dir/L"nvinfer_plugin_10.dll"),cuda(dir/L"cudart64_13.dll");Log logger;
    using Init=bool(__cdecl*)(void*,const char*);need(plugins.get<Init>("initLibNvInferPlugins")(&logger,""),"plugins");
    using Create=void*(__cdecl*)(void*,int);std::unique_ptr<IRuntime> runtime(static_cast<IRuntime*>(infer.get<Create>("createInferRuntime_INTERNAL")(&logger,NV_TENSORRT_VERSION)));need(bool(runtime),"runtime");
#define FN(n) auto n=cuda.get<decltype(&::n)>(#n)
    FN(cudaSetDevice);FN(cudaMalloc);FN(cudaFree);FN(cudaMemcpy);FN(cudaStreamCreate);FN(cudaStreamDestroy);FN(cudaStreamSynchronize);
#undef FN
    need(cudaSetDevice(0)==cudaSuccess,"device");Plan baseline(*runtime,argv[2]),head(*runtime,argv[3]),tail(*runtime,argv[4]);
    Dims dims{};dims.nbDims=4;dims.d[0]=2;dims.d[1]=4;dims.d[2]=544;dims.d[3]=960;
    need(tail.context->setInputShape("features",dims),"feature shape");
    constexpr size_t N=960*540,F=2*4*544*960;void* input=nullptr;void* features=nullptr;void* a=nullptr;void* b=nullptr;
    need(cudaMalloc(&input,N*7*4)==cudaSuccess&&cudaMalloc(&features,F*4)==cudaSuccess&&cudaMalloc(&a,N*5*4)==cudaSuccess&&cudaMalloc(&b,N*5*4)==cudaSuccess,"buffers");
    baseline.bind("input",input);baseline.bind("motion",a);head.bind("input",input);head.bind("features",features);tail.bind("input",input);tail.bind("features",features);tail.bind("motion",b);
    cudaStream_t stream=nullptr;need(cudaStreamCreate(&stream)==cudaSuccess,"stream");
    std::vector<float> host(N*7),left(N*5),right(N*5),featureFirst(F),featureLast(F);
    uint64_t different=0,total=0,phasesDifferent=0;double maxAbs=0,sumAbs=0;bool headIndependent=true;
    std::ofstream report(argv[5]);report<<"{\n  \"scope\": \"offline trained engines; pair-local Head reuse; FP32 feature boundary; no D3D/player cache\",\n  \"comparisons\": [\n";
    unsigned comparison=0;
    for(unsigned pair=0;pair<8;++pair){
        uint32_t rng=12345+pair;
        for(size_t i=0;i<N*6;++i){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;host[i]=pair%2?float(rng&65535)/65535.0f:float((i+pair*17)%997)/996.0f;}
        std::fill(host.begin()+N*6,host.end(),.2f);
        need(cudaMemcpy(input,host.data(),host.size()*4,cudaMemcpyHostToDevice)==cudaSuccess,"input");head.run(stream);need(cudaStreamSynchronize(stream)==cudaSuccess,"head ready");
        need(cudaMemcpy(featureFirst.data(),features,F*4,cudaMemcpyDeviceToHost)==cudaSuccess,"features first");
        for(unsigned phase=1;phase<=4;++phase){
            std::fill(host.begin()+N*6,host.end(),float(phase)/5);
            need(cudaMemcpy(input,host.data(),host.size()*4,cudaMemcpyHostToDevice)==cudaSuccess,"phase input");baseline.run(stream);tail.run(stream);need(cudaStreamSynchronize(stream)==cudaSuccess,"results ready");
            need(cudaMemcpy(left.data(),a,N*5*4,cudaMemcpyDeviceToHost)==cudaSuccess&&cudaMemcpy(right.data(),b,N*5*4,cudaMemcpyDeviceToHost)==cudaSuccess,"read results");
            uint64_t count=0;double localMax=0;
            for(size_t i=0;i<left.size();++i){need(std::isfinite(left[i])&&std::isfinite(right[i]),"finite output");if(std::memcmp(&left[i],&right[i],4)){++count;const double delta=std::abs(double(left[i])-right[i]);sumAbs+=delta;localMax=(std::max)(localMax,delta);}}
            total+=left.size();different+=count;phasesDifferent+=count!=0;maxAbs=(std::max)(maxAbs,localMax);
            if(comparison++)report<<",\n";report<<"    {\"pair\": "<<pair<<", \"phase\": "<<phase<<", \"different_floats\": "<<count<<", \"max_abs\": "<<localMax<<"}";
        }
        head.run(stream);need(cudaStreamSynchronize(stream)==cudaSuccess,"head last ready");need(cudaMemcpy(featureLast.data(),features,F*4,cudaMemcpyDeviceToHost)==cudaSuccess,"features last");headIndependent=headIndependent&&(featureFirst==featureLast);
    }
    report<<"\n  ],\n  \"head_t_independent\": "<<(headIndependent?"true":"false")<<",\n  \"compared_floats\": "<<total<<",\n  \"different_floats\": "<<different<<",\n  \"different_phases\": "<<phasesDifferent<<",\n  \"max_abs\": "<<maxAbs<<",\n  \"mean_abs\": "<<sumAbs/total<<",\n  \"decision\": \""<<(different?"HOLD_SPLIT_NUMERICS_DIFFER":"NUMERIC_PASS_COST_AND_FINAL_PIXELS_PENDING")<<"\"\n}\n";
    cudaFree(b);cudaFree(a);cudaFree(features);cudaFree(input);cudaStreamDestroy(stream);
    std::cout<<"COMPLETED comparisons="<<comparison<<" differentPhases="<<phasesDifferent<<" differentFloats="<<different<<" headIndependent="<<headIndependent<<" maxAbs="<<maxAbs<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
