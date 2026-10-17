// Experimental pair-local reuse only. No image/PTS/texture-address cache survives
// render_batch(). The owner serializes work on its CUDA stream and drains that
// stream before releasing this allocation, including reset/seek/error teardown.
std::atomic<uint64_t> head_pairs{0},head_phases{0},head_plan_loads{0},head_context_creates{0};
std::atomic<uint64_t> head_preparation_jobs{0};
struct RifeHeadContexts {
    std::unique_ptr<nvinfer1::IExecutionContext> head,tail;
};
struct RifeHeadPlans {
    std::shared_ptr<SharedEngine> base;
    std::unique_ptr<nvinfer1::ICudaEngine> head,tail;
    std::mutex mutex;
    RifeHeadContexts idle;
    static constexpr size_t FeatureElements=size_t(2)*4*544*GW;
    RifeHeadPlans(std::shared_ptr<SharedEngine> engine,const std::filesystem::path& directory):base(std::move(engine)){
        auto load=[&](const wchar_t* name){
            std::ifstream f(directory/name,std::ios::binary|std::ios::ate);
            if(!f)throw std::runtime_error("Optional RIFE Head cache engine is missing");
            const auto size=f.tellg();if(size<1024||size>1024ll*1024*1024)throw std::runtime_error("Invalid Head cache engine size");
            std::vector<char> bytes(static_cast<size_t>(size));f.seekg(0);f.read(bytes.data(),bytes.size());
            if(!f)throw std::runtime_error("Cannot read Head cache engine");
            auto result=std::unique_ptr<nvinfer1::ICudaEngine>(base->runtime->deserializeCudaEngine(bytes.data(),bytes.size()));
            if(!result)throw std::runtime_error("Incompatible Head cache engine");return result;
        };
        head=load(L"rife425_head_960_fp16.engine");tail=load(L"rife425_tail_960_fp16.engine");
        auto shape=[](nvinfer1::ICudaEngine& e,const char* name,int n,int c,int h,int w){
            const auto s=e.getTensorShape(name);
            return s.nbDims==4&&s.d[0]==n&&s.d[1]==c&&s.d[2]==h&&s.d[3]==w&&e.getTensorDataType(name)==nvinfer1::DataType::kFLOAT;
        };
        if(head->getNbIOTensors()!=2||tail->getNbIOTensors()!=3||
           !shape(*head,"input",1,7,GH,GW)||!shape(*head,"features",2,4,544,GW)||
           !shape(*tail,"input",1,7,GH,GW)||!shape(*tail,"features",2,4,544,GW)||!shape(*tail,"motion",1,5,GH,GW))
            throw std::runtime_error("Head cache tensor contract mismatch");
        ++head_plan_loads;
    }
    RifeHeadContexts take(){
        std::lock_guard<std::mutex> lock(mutex);
        if(idle.head&&idle.tail&&!environment_flag(L"RIFE_DISABLE_CONTEXT_REUSE"))return std::move(idle);
        RifeHeadContexts result{std::unique_ptr<nvinfer1::IExecutionContext>(head->createExecutionContext()),std::unique_ptr<nvinfer1::IExecutionContext>(tail->createExecutionContext())};
        if(!result.head||!result.tail)throw std::runtime_error("Cannot allocate Head cache contexts");
        ++head_context_creates;return result;
    }
    void recycle(RifeHeadContexts& contexts)noexcept{
        if(!contexts.head||!contexts.tail||environment_flag(L"RIFE_DISABLE_CONTEXT_REUSE"))return;
        if(!contexts.head->setTensorAddress("input",nullptr)||!contexts.head->setTensorAddress("features",nullptr)||
           !contexts.tail->setTensorAddress("input",nullptr)||!contexts.tail->setTensorAddress("features",nullptr)||!contexts.tail->setTensorAddress("motion",nullptr))return;
        try{std::lock_guard<std::mutex> lock(mutex);if(!idle.head&&!idle.tail)idle=std::move(contexts);}catch(...){}
    }
    ~RifeHeadPlans(){base->cuda->cudaSetDevice(base->device);idle={};tail.reset();head.reset();}
};
std::mutex& head_plans_mutex(){static std::mutex mutex;return mutex;}
std::map<std::wstring,std::shared_ptr<RifeHeadPlans>>& head_plans_cache(){static std::map<std::wstring,std::shared_ptr<RifeHeadPlans>> plans;return plans;}
std::wstring head_plans_key(const std::shared_ptr<SharedEngine>& base,const std::filesystem::path& runtime){
    const auto directory=std::filesystem::absolute(runtime).lexically_normal().parent_path()/L"engines";
    return directory.wstring()+L"/"+std::to_wstring(base->device);
}
std::shared_ptr<RifeHeadPlans> ready_head_plans(const std::shared_ptr<SharedEngine>& base,const std::filesystem::path& runtime){
    std::unique_lock<std::mutex> lock(head_plans_mutex(),std::try_to_lock);if(!lock)return {};
    const auto& plans=head_plans_cache();const auto it=plans.find(head_plans_key(base,runtime));return it==plans.end()?nullptr:it->second;
}
std::shared_ptr<RifeHeadPlans> cached_head_plans(const std::shared_ptr<SharedEngine>& base,const std::filesystem::path& runtime){
    const auto directory=std::filesystem::absolute(runtime).lexically_normal().parent_path()/L"engines";
    std::lock_guard<std::mutex> lock(head_plans_mutex());
    auto& p=head_plans_cache()[head_plans_key(base,runtime)];if(!p)p=std::make_shared<RifeHeadPlans>(base,directory);return p;
}
struct RifeHeadPreparation {
    std::shared_ptr<SharedEngine> base;
    std::filesystem::path runtime;
    std::mutex mutex;
    std::shared_ptr<RifeHeadPlans> ready;
    bool failed=false;
    HMODULE module=nullptr;
};
DWORD WINAPI prepare_head_entry(void* argument){
    HMODULE module=nullptr;
    {
        std::unique_ptr<std::shared_ptr<RifeHeadPreparation>> holder(static_cast<std::shared_ptr<RifeHeadPreparation>*>(argument));
        auto request=*holder;module=request->module;
        try{
            auto& cu=*request->base->cuda;cu.check(cu.cudaSetDevice(request->base->device),"Select Head preparation GPU");
            auto plans=cached_head_plans(request->base,request->runtime);
            auto contexts=plans->take();plans->recycle(contexts);
            std::lock_guard<std::mutex> lock(request->mutex);request->ready=std::move(plans);
        }catch(const std::exception& error){
            OutputDebugStringA((std::string("RIFE optional Head preparation unavailable: ")+error.what()+"\n").c_str());
            std::lock_guard<std::mutex> lock(request->mutex);request->failed=true;
        }catch(...){std::lock_guard<std::mutex> lock(request->mutex);request->failed=true;}
    }
    --head_preparation_jobs;
    FreeLibraryAndExitThread(module,0);
}
std::shared_ptr<RifeHeadPreparation> request_head_preparation(const std::shared_ptr<SharedEngine>& base,const std::filesystem::path& runtime){
    // At most one immutable-plan/context preparation job per package/GPU.
    // It has no image state and cannot publish a result into a later pair.
    static std::mutex mutex;
    static std::map<std::wstring,std::shared_ptr<RifeHeadPreparation>> requests;
    const auto key=std::filesystem::absolute(runtime).lexically_normal().wstring()+L"/"+std::to_wstring(base->device);
    std::lock_guard<std::mutex> lock(mutex);
    auto& result=requests[key];if(result)return result;
    result=std::make_shared<RifeHeadPreparation>();result->base=base;result->runtime=runtime;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,reinterpret_cast<LPCWSTR>(&prepare_head_entry),&result->module)){result->failed=true;return result;}
    auto argument=std::make_unique<std::shared_ptr<RifeHeadPreparation>>(result);
    ++head_preparation_jobs;
    HANDLE thread=CreateThread(nullptr,0,prepare_head_entry,argument.get(),0,nullptr);
    if(!thread){--head_preparation_jobs;FreeLibrary(result->module);result->failed=true;return result;}
    argument.release();CloseHandle(thread);return result;
}
struct RifePairHeadCache {
    std::shared_ptr<RifeHeadPlans> plans;
    RifeHeadContexts contexts;
    RifeGraph head_graph,tail_graph;
    void* features=nullptr;
    bool recycle=false;
    explicit RifePairHeadCache(std::shared_ptr<RifeHeadPlans> value):plans(std::move(value)){}
    void initialize(){
        auto& cu=*plans->base->cuda;contexts=plans->take();
        cu.check(cu.cudaMalloc(&features,RifeHeadPlans::FeatureElements*sizeof(float)),"Allocate pair-local Head features");
        nvinfer1::Dims input{};input.nbDims=4;input.d[0]=1;input.d[1]=7;input.d[2]=GH;input.d[3]=GW;
        nvinfer1::Dims feature{};feature.nbDims=4;feature.d[0]=2;feature.d[1]=4;feature.d[2]=544;feature.d[3]=GW;
        if(!contexts.head->setInputShape("input",input)||!contexts.tail->setInputShape("input",input)||
           !contexts.tail->setInputShape("features",feature)||!contexts.tail->setTensorAddress("features",features))
            throw std::runtime_error("Cannot bind pair-local Head features");
    }
    // Adapt only the output name; RifeGraph retains its existing changed-address
    // synchronization, fixed capture, and capture-failure fallback contracts.
    struct HeadAdapter {
        nvinfer1::IExecutionContext& context;
        bool setTensorAddress(const char* name,void* address){return context.setTensorAddress(std::strcmp(name,"motion")==0?"features":name,address);}
        bool enqueueV3(cudaStream_t stream){return context.enqueueV3(stream);}
    };
    void begin_pair(Cuda& cu,cudaStream_t stream,void* input,bool graph_enabled){
        HeadAdapter adapter{*contexts.head};
        head_graph.run(cu,adapter,stream,{input,features},graph_enabled);++head_pairs;
    }
    void phase(Cuda& cu,cudaStream_t stream,const std::array<void*,2>& pointers,bool graph_enabled){
        tail_graph.run(cu,*contexts.tail,stream,pointers,graph_enabled);++head_phases;
    }
    ~RifePairHeadCache(){
        auto& cu=*plans->base->cuda;cu.cudaSetDevice(plans->base->device);
        head_graph.clear(cu);tail_graph.clear(cu);
        if(recycle)plans->recycle(contexts);
        contexts={};if(features)cu.cudaFree(features);
    }
};
