#define wmain unused_retirement_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"

int wmain(int argc,wchar_t** argv){try{
    require(argc==2,"rife_retirement_tests runtime");
    SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",L"1");
    SetEnvironmentVariableW(L"RIFE_HEAD_CACHE_SYNC_PREPARE",nullptr);
    Device dev;auto a=dev.upload(pattern(1920,1080,0,true),1920,1080,0,true);
    auto b=dev.upload(pattern(1920,1080,8,true),1920,1080,400000,true);
    const std::vector<int64_t> times{80000,160000,240000,320000};
    auto e=engine(dev,argv[1]);auto retained=e->interpolate_pair(a,b,times);
    std::vector<std::string> reference;for(auto& f:retained.frames)reference.push_back(hash_bytes(dev.read(f)));
    require(MotionSynthesizer::optional_preload_pending()!=0,"cold optional preparation already finished");
    auto host=std::make_shared<int>(1);std::weak_ptr<int> lifetime=host;
    const auto start=Clock::now();
    require(GpuRifeEngine::retire_during_preparation(e,host),"retirement not accepted");
    require(!e,"caller retained retired engine");host.reset();
    const auto elapsedMs=elapsed(start);require(elapsedMs<100,"retirement blocked caller");
    if(GpuRifeEngine::retirement_pending())require(!lifetime.expired(),"borrowed host lifetime released too soon");
    const auto contexts=MotionSynthesizer::execution_create_count();
    std::atomic<bool> cancelled{false};
    auto resumed=engine(dev,argv[1]);
    require(GpuRifeEngine::retirement_pending()!=0,"cold retirement finished before warmup probe");
    const auto warmupStart=Clock::now();const auto warmup=resumed->startup_warmup();
    const auto warmupMs=elapsed(warmupStart);
    std::cout<<"RETIREMENT_WARMUP result="<<warmup<<" ms="<<warmupMs<<std::endl;
    require(warmup==-1&&warmupMs<100,"startup warmup waited on unfinished retirement/optional loader");
    auto next=resumed->interpolate_pair(a,b,times,&cancelled);
    require(!GpuRifeEngine::retirement_pending()&&lifetime.expired(),"retirement did not release ownership");
    require(MotionSynthesizer::execution_create_count()==contexts,"retirement lost reusable full context");
    for(size_t i=0;i<reference.size();++i){
        require(hash_bytes(dev.read(retained.frames[i]))==reference[i],"retired output changed");
        require(hash_bytes(dev.read(next.frames[i]))==reference[i],"resumed output changed");
    }
    std::cout<<"PASS retireMs="<<elapsedMs<<" hostLifetime=1 retainedOutputs=4 exactResume=4 reusedContext=1\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
