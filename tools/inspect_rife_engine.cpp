// Offline inspection/profiling of a copied engine. Never loaded by the filter.
#include <windows.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <vector>
#include <stdexcept>
using namespace nvinfer1;namespace fs=std::filesystem;
void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Lib {
    HMODULE h;explicit Lib(const fs::path& p):h(LoadLibraryExW(p.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)){need(h!=nullptr,"load runtime");}
    ~Lib(){if(h)FreeLibrary(h);}template<class T>T get(const char* n){auto f=GetProcAddress(h,n);need(f!=nullptr,n);return reinterpret_cast<T>(f);}
};
struct Log:ILogger {void log(Severity s,const char* t)noexcept override{if(s<=Severity::kWARNING)std::cerr<<t<<'\n';}};
struct Profile:IProfiler {
    std::map<std::string,std::pair<double,unsigned>> rows;
    void reportLayerTime(const char* name,float ms)noexcept override{try{auto& r=rows[name];r.first+=ms;++r.second;}catch(...){}}
};
int wmain(int argc,wchar_t** argv){try {
    need(argc==5,"runtime engine layers.json profile.tsv");fs::path dir=fs::absolute(argv[1]);
    Lib infer(dir/L"nvinfer_10.dll"),plugins(dir/L"nvinfer_plugin_10.dll"),cuda(dir/L"cudart64_13.dll");Log logger;
    using Init=bool(__cdecl*)(void*,const char*);need(plugins.get<Init>("initLibNvInferPlugins")(&logger,""),"plugins");
    using Create=void*(__cdecl*)(void*,int);std::unique_ptr<IRuntime> runtime(static_cast<IRuntime*>(infer.get<Create>("createInferRuntime_INTERNAL")(&logger,NV_TENSORRT_VERSION)));
    std::ifstream file(argv[2],std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
    need(runtime&&!bytes.empty(),"runtime/engine");std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(),bytes.size()));need(bool(engine),"deserialize");
    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());need(bool(context),"context");
    Dims shape{};shape.nbDims=4;shape.d[0]=1;shape.d[1]=7;shape.d[2]=540;shape.d[3]=960;need(context->setInputShape("input",shape),"shape");
    std::unique_ptr<IEngineInspector> inspector(engine->createEngineInspector());need(bool(inspector),"inspector");inspector->setExecutionContext(context.get());
    std::ofstream layers(argv[3]);layers<<inspector->getEngineInformation(LayerInformationFormat::kJSON);layers.close();
#define FN(n) auto n=cuda.get<decltype(&::n)>(#n)
    FN(cudaSetDevice);FN(cudaMalloc);FN(cudaFree);FN(cudaMemcpy);FN(cudaStreamCreate);FN(cudaStreamDestroy);FN(cudaStreamSynchronize);
#undef FN
    need(cudaSetDevice(0)==cudaSuccess,"device");void* input=nullptr;void* output=nullptr;constexpr size_t N=960*540;
    need(cudaMalloc(&input,N*7*4)==cudaSuccess&&cudaMalloc(&output,N*5*4)==cudaSuccess,"buffers");
    std::vector<float> host(N*7,.5f);need(cudaMemcpy(input,host.data(),host.size()*4,cudaMemcpyHostToDevice)==cudaSuccess,"input copy");
    need(context->setTensorAddress("input",input)&&context->setTensorAddress("motion",output),"bindings");
    cudaStream_t stream=nullptr;need(cudaStreamCreate(&stream)==cudaSuccess,"stream");Profile profile;context->setProfiler(&profile);
    for(int i=0;i<35;++i){need(context->enqueueV3(stream),"enqueue");need(cudaStreamSynchronize(stream)==cudaSuccess,"complete");if(i==4)profile.rows.clear();}
    std::ofstream csv(argv[4]);csv<<"layer\tcount\tmean_ms\n";
    double sum=0;for(auto& [name,value]:profile.rows){csv<<name<<'\t'<<value.second<<'\t'<<value.first/value.second<<'\n';sum+=value.first/value.second;}
    context->setProfiler(nullptr);cudaFree(output);cudaFree(input);cudaStreamDestroy(stream);
    std::cout<<"PASS layers="<<profile.rows.size()<<" sum_profiled_layer_ms="<<sum<<" scope=isolated_TRT_no_CUDA_Graph_no_decode_or_composition\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
