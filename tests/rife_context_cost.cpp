#include <windows.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <future>
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
struct Logger:nvinfer1::ILogger{void log(Severity s,const char* m)noexcept override{if(s<=Severity::kWARNING)std::cerr<<m<<'\n';}};
int wmain(int argc,wchar_t** argv){try{
 if(argc<3)throw std::runtime_error("runtime static|user|async [plan]");
 std::filesystem::path dir=argv[1];const std::wstring mode=argv[2];
 auto cu=LoadLibraryExW((dir/L"cudart64_13.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
 auto trt=LoadLibraryExW((dir/L"nvinfer_10.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
 if(!cu||!trt)throw std::runtime_error("runtime load");
#define API(name) auto name=reinterpret_cast<decltype(&::name)>(GetProcAddress(cu,#name));if(!name)throw std::runtime_error(#name)
 API(cudaSetDevice);API(cudaMalloc);API(cudaMallocAsync);API(cudaFree);API(cudaStreamCreateWithFlags);API(cudaStreamSynchronize);API(cudaStreamDestroy);
 auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error("CUDA "+std::to_string(int(e)));};
 auto start=Clock::now();check(cudaSetDevice(0));std::cout<<"deviceMs="<<ms(start)<<'\n';
 Logger logger;using Create=void*(*)(void*,int32_t);
 std::unique_ptr<nvinfer1::IRuntime> runtime(static_cast<nvinfer1::IRuntime*>(reinterpret_cast<Create>(GetProcAddress(trt,"createInferRuntime_INTERNAL"))(&logger,NV_TENSORRT_VERSION)));
 std::ifstream file(dir.parent_path()/L"engines"/(argc>3?argv[3]:L"rife425_standard_960_fp16.engine"),std::ios::binary|std::ios::ate);
 if(!file)throw std::runtime_error("engine file");std::vector<char> data(size_t(file.tellg()));file.seekg(0);file.read(data.data(),data.size());
 start=Clock::now();std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(data.data(),data.size()));if(!engine)throw std::runtime_error("engine deserialize");std::cout<<"engineMs="<<ms(start)<<'\n';
 if(mode==L"parallel"){
  std::ifstream other(dir.parent_path()/L"engines/rife425_tail_960_fp16.engine",std::ios::binary|std::ios::ate);
  if(!other)throw std::runtime_error("tail file");std::vector<char> bytes(size_t(other.tellg()));other.seekg(0);other.read(bytes.data(),bytes.size());
  std::unique_ptr<nvinfer1::ICudaEngine> tail(runtime->deserializeCudaEngine(bytes.data(),bytes.size()));if(!tail)throw std::runtime_error("tail deserialize");
  start=Clock::now();
  auto pending=std::async(std::launch::async,[&]{check(cudaSetDevice(0));auto tick=Clock::now();std::unique_ptr<nvinfer1::IExecutionContext> ctx(tail->createExecutionContext());if(!ctx)throw std::runtime_error("tail context");std::cout<<"tailMs="<<ms(tick)<<std::endl;return ctx;});
  std::unique_ptr<nvinfer1::IExecutionContext> full(engine->createExecutionContext());if(!full)throw std::runtime_error("full context");std::cout<<"fullMs="<<ms(start)<<std::endl;
  auto tailContext=pending.get();std::cout<<"PARALLEL wallMs="<<ms(start)<<std::endl;return 0;
 }
 for(int i=0;i<3;++i){
  start=Clock::now();std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext(mode==L"static"?nvinfer1::ExecutionContextAllocationStrategy::kSTATIC:nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if(!context)throw std::runtime_error("context");double contextMs=ms(start),allocMs=0;void* memory=nullptr;cudaStream_t stream=nullptr;const auto bytes=engine->getDeviceMemorySizeV2();
  if(mode!=L"static"){
   start=Clock::now();
   if(mode==L"async"){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaMallocAsync(&memory,size_t(bytes),stream));check(cudaStreamSynchronize(stream));}
   else check(cudaMalloc(&memory,size_t(bytes)));
   context->setDeviceMemoryV2(memory,bytes);allocMs=ms(start);
  }
  std::cout<<"CONTEXT index="<<i<<" createMs="<<contextMs<<" allocationMs="<<allocMs<<" bytes="<<bytes<<std::endl;
  context.reset();if(memory)check(cudaFree(memory));if(stream)check(cudaStreamDestroy(stream));
 }
 engine.reset();runtime.reset();FreeLibrary(trt);FreeLibrary(cu);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
