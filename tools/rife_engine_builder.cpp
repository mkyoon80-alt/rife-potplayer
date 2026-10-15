// RIFE-specific offline engine preparation. Never loaded by the playback filter.
#include <windows.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>
namespace fs=std::filesystem;
using namespace nvinfer1;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Library {
 HMODULE h;
 explicit Library(const fs::path& p):h(LoadLibraryExW(p.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)){require(h!=nullptr,"TensorRT dependency could not be loaded");}
 ~Library(){if(h)FreeLibrary(h);}
 template<class T>T get(const char* name){auto f=GetProcAddress(h,name);require(f!=nullptr,"TensorRT API symbol missing");return reinterpret_cast<T>(f);}
};
struct Logger:ILogger {
 void log(Severity s,const char* text)noexcept override{if(s<=Severity::kINFO)std::cerr<<"[TensorRT "<<int(s)<<"] "<<text<<std::endl;}
};
std::vector<char> read(const fs::path& path){std::ifstream f(path,std::ios::binary);return std::vector<char>((std::istreambuf_iterator<char>(f)),{});}
void write(const fs::path& path,const void* bytes,size_t size){std::ofstream f(path,std::ios::binary|std::ios::trunc);require(bool(f),"Cannot open output file");f.write(static_cast<const char*>(bytes),size);f.close();require(bool(f),"Cannot finish output file");}
int wmain(int argc,wchar_t** argv){try{
 require(argc==7,"Usage: RifeEngineBuilder --build|--inspect runtime-dir builder-dir model.onnx output.engine timing.cache");
 const bool inspect=std::wstring(argv[1])==L"--inspect";
 require(inspect||std::wstring(argv[1])==L"--build","Unknown action");
 fs::path runtime=fs::absolute(argv[2]), resources=fs::absolute(argv[3]),model=fs::absolute(argv[4]);
 SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS|LOAD_LIBRARY_SEARCH_USER_DIRS);
 auto runtimeCookie=AddDllDirectory(runtime.c_str());auto resourceCookie=AddDllDirectory(resources.c_str());
 require(runtimeCookie&&resourceCookie,"Cannot configure local runtime search paths");
 Logger logger;
 Library infer(runtime/L"nvinfer_10.dll"),plugins(runtime/L"nvinfer_plugin_10.dll"),parserLib(resources/L"nvonnxparser_10.dll");
 using Init=bool(__cdecl*)(void*,const char*);
 require(plugins.get<Init>("initLibNvInferPlugins")(&logger,""),"Cannot initialize TensorRT plugins");
 using Create=void*(__cdecl*)(void*,int);
 std::unique_ptr<IBuilder> builder(static_cast<IBuilder*>(infer.get<Create>("createInferBuilder_INTERNAL")(&logger,NV_TENSORRT_VERSION)));
 require(bool(builder),"Cannot create TensorRT builder for this GPU/runtime");
 std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
 require(bool(network),"Cannot create network");
 using CreateParser=void*(__cdecl*)(void*,void*,int);
 std::unique_ptr<nvonnxparser::IParser> parser(static_cast<nvonnxparser::IParser*>(parserLib.get<CreateParser>("createNvOnnxParser_INTERNAL")(network.get(),&logger,NV_ONNX_PARSER_VERSION)));
 require(bool(parser),"Cannot create ONNX parser");
 auto onnx=read(model);require(!onnx.empty(),"ONNX model missing or empty");
 require(parser->parse(onnx.data(),onnx.size()),"Cannot parse the RIFE ONNX model");
 require(network->getNbInputs()==1&&network->getNbOutputs()==1,"Unexpected RIFE model interface");
 auto input=network->getInput(0);auto output=network->getOutput(0);
 require(std::string(input->getName())=="input"&&std::string(output->getName())=="motion","Unexpected RIFE tensor names");
 std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());require(bool(config),"Cannot create builder configuration");
 config->setFlag(BuilderFlag::kFP16);config->clearFlag(BuilderFlag::kTF32);config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
 config->setBuilderOptimizationLevel(3);config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
 const std::set<std::string> fp32={"/Cast_2","/Cast_3","/Cast_5","/Cast_7","/Reciprocal","/Reciprocal_1","/Mul","/Mul_1","/Mul_8","/Mul_10","/Sub_5","/Sub_6","/Sigmoid","/Add_7","lab_concat"};
 std::set<std::string> found;int constrained=0;
 auto nonActivation=[](DataType t){return t==DataType::kINT32||t==DataType::kINT64||t==DataType::kBOOL||t==DataType::kUINT8;};
 for(int i=0;i<network->getNbLayers();++i){
  auto layer=network->getLayer(i);std::string name=layer->getName();const bool exact=fp32.count(name)!=0;
  if(!exact&&name.rfind("ONNXTRT_Broadcast_",0)!=0)continue;
  // Match the official trtexec 10.16 wildcard rules: shape/index constants
  // are not floating-point activations. Exact named constraints stay strict.
  if(!exact){
   if(nonActivation(layer->getPrecision()))continue;
   if(layer->getType()==LayerType::kCONSTANT&&nonActivation(static_cast<IConstantLayer*>(layer)->getWeights().type))continue;
   if(layer->getNbInputs()>0&&layer->getInput(0)->isShapeTensor())continue;
   if(layer->getNbInputs()>0&&layer->getNbOutputs()>0&&nonActivation(layer->getInput(0)->getType())&&nonActivation(layer->getOutput(0)->getType()))continue;
  }
  layer->setPrecision(DataType::kFLOAT);++constrained;if(exact)found.insert(name);
 }
 require(found==fp32,"RIFE precision-critical layers changed; preparation stopped");
 for(auto tensor:{input,output}){tensor->setType(DataType::kFLOAT);tensor->setAllowedFormats(1U<<int(TensorFormat::kLINEAR));}
 auto profile=builder->createOptimizationProfile();require(profile!=nullptr,"Cannot create input profile");
 Dims dims{};dims.nbDims=4;dims.d[0]=1;dims.d[1]=7;dims.d[2]=540;dims.d[3]=960;
 for(auto selector:{OptProfileSelector::kMIN,OptProfileSelector::kOPT,OptProfileSelector::kMAX})require(profile->setDimensions(input->getName(),selector,dims),"Cannot set fixed 960 input shape");
 require(profile->isValid()&&config->addOptimizationProfile(profile)>=0,"Invalid input profile");
 auto cacheBytes=read(argv[6]);std::unique_ptr<ITimingCache> cache(config->createTimingCache(cacheBytes.empty()?nullptr:cacheBytes.data(),cacheBytes.size()));
 require(bool(cache),"Cannot create timing cache");
 if(!config->setTimingCache(*cache,false)){cache.reset(config->createTimingCache(nullptr,0));require(cache&&config->setTimingCache(*cache,false),"Cannot reset incompatible timing cache");}
 std::cout<<"RIFE 4.25 Standard: fixed input 1x7x540x960; FP16; TF32 off; FP32 critical layers="<<constrained<<"; FP32 linear I/O; optimization=3"<<std::endl;
 if(inspect){std::cout<<"PASS configuration inspection"<<std::endl;return 0;}
 std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network,*config));require(bool(plan),"Engine build failed");
 // A candidate must deserialize and allocate an execution context before it
 // can be handed to the preparation script for full D3D11 inference testing.
 std::unique_ptr<IRuntime> checkRuntime(static_cast<IRuntime*>(infer.get<Create>("createInferRuntime_INTERNAL")(&logger,NV_TENSORRT_VERSION)));
 require(bool(checkRuntime),"Cannot create validation runtime");
 std::unique_ptr<ICudaEngine> engine(checkRuntime->deserializeCudaEngine(plan->data(),plan->size()));require(bool(engine),"Built engine failed deserialization");
 std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());require(bool(context),"Built engine cannot allocate an execution context");
 require(context->setInputShape("input",dims),"Built engine rejected the fixed input shape");
 write(argv[5],plan->data(),plan->size());
 std::unique_ptr<IHostMemory> serializedCache(config->getTimingCache()->serialize());if(serializedCache)write(argv[6],serializedCache->data(),serializedCache->size());
 std::unique_ptr<IEngineInspector> inspector(engine->createEngineInspector());
 if(inspector){const char* info=inspector->getEngineInformation(LayerInformationFormat::kJSON);if(info)write(fs::path(argv[5]).wstring()+L".layers.json",info,strlen(info));}
 std::cout<<"PASS candidate engine prepared; run D3D11 validation before playback"<<std::endl;return 0;
 }catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<std::endl;return 1;}}
