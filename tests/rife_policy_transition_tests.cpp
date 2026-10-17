#define wmain unused_policy_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"
#include "nvof/rife_optimization_policy.hpp"

int wmain(int argc,wchar_t** argv){timeBeginPeriod(1);try{
    require(argc==2,"runtime required");Device dev;
    SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",L"1");
    SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PACKED_MOTION",L"1");
    auto candidate=engine(dev,argv[1]);candidate->enable_timing(true,true,1);
    const auto began=Clock::now();
    const auto real_heads=MotionSynthesizer::head_cache_pair_count();
    unsigned submissions=0;
    while(!candidate->startup_ready()){
        const int submitted=candidate->startup_warmup(1920,1080,true);
        if(submitted>0)submissions+=unsigned(submitted);
        require(elapsed(began)<15000,"selected runtime warmup timeout");Sleep(1);
    }
    require(submissions==2&&MotionSynthesizer::head_cache_load_count()==0,"optional split loading delayed cold startup");
    std::cout<<"COLD_READY full-only=1 optionalPlanLoads=0"<<std::endl;
    SetEnvironmentVariableW(L"RIFE_HEAD_CACHE_SYNC_PREPARE",L"1");
    do{
        const int submitted=candidate->startup_warmup(1920,1080,true);
        if(submitted>0)submissions+=unsigned(submitted);
        require(elapsed(began)<15000,"cached split warmup timeout");Sleep(1);
    }while(!candidate->startup_ready());
    SetEnvironmentVariableW(L"RIFE_HEAD_CACHE_SYNC_PREPARE",nullptr);
    require(submissions==6,"full and split graph warmup was not submitted");
    require(MotionSynthesizer::head_cache_pair_count()==real_heads,"dummy Head counted as real source pair");
    std::cout<<"WARMUP gpuReady=1 enqueues="<<submissions<<" elapsedMs="<<elapsed(began)<<std::endl;
    size_t checked=0;
    for(auto size:std::vector<std::array<int,2>>{{1920,1080},{2560,1440},{3840,1604},{3840,2160},{7680,4320},{2560,1440},{1920,1080},{1604,3840}}){
        const int w=size[0],h=size[1];const bool packed=rife_packed_motion_resolution(w,h);
        const unsigned transfer=packed?15:0;
        auto a=dev.upload(pattern(w,h,0,true),w,h,0,true,transfer);
        auto b=dev.upload(pattern(w,h,8,true),w,h,400000,true,transfer);
        auto metadata=std::make_shared<HdrMetadata>();metadata->data[0].assign(80,17);a.hdr=metadata;
        const std::vector<int64_t> times{100000,200000,300000};
        SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",L"0");
        SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PACKED_MOTION",L"0");
        auto original=engine(dev,argv[1]);auto expected=original->interpolate_pair(a,b,times);
        candidate->reset();
        const auto heads=MotionSynthesizer::head_cache_pair_count();
        auto actual=candidate->interpolate_pair(a,b,times);dev.wait();
        require(MotionSynthesizer::head_cache_pair_count()==heads+1,"new source pair failed to recompute Head");
        require(actual.frames.size()==times.size()&&expected.frames.size()==times.size(),"missing interpolated frames");
        auto timing=candidate->pair_timing();require(timing.rife&&timing.rife->phase_count==3,"path timing absent");
        for(unsigned i=0;i<3;++i){
            const auto& phase=timing.rife->phases[i];
            require(phase.packed_layout==packed&&phase.head_cached,"incorrect E03/E04 selection after source size change");
            require(actual.frames[i].pts==times[i]&&actual.frames[i].hdr==metadata&&actual.frames[i].hdr_transfer==transfer,"output PTS/HDR changed");
            require(hash_bytes(dev.read(actual.frames[i]))==hash_bytes(dev.read(expected.frames[i])),"resolution policy changed final pixels");++checked;
        }
        const auto before_single=MotionSynthesizer::head_cache_pair_count();
        auto single=candidate->interpolate_pair(a,b,{200000});
        require(MotionSynthesizer::head_cache_pair_count()==before_single,"one output used Head cache");
        require(hash_bytes(dev.read(single.frames[0]))==hash_bytes(dev.read(expected.frames[1])),"single output changed pixels");++checked;
        std::cout<<"PASS source="<<w<<'x'<<h<<" packed="<<packed<<" multiHead=1 singleHead=0 exactOutputs=4"<<std::endl;
    }
    std::cout<<"PASS totalExactOutputs="<<checked<<" same-engine-resolution-switch=1 dummy-cache-invalid=1\n";
    timeEndPeriod(1);return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;timeEndPeriod(1);return 1;}}
