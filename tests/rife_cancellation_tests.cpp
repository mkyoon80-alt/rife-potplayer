#define wmain unused_cancellation_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"

int wmain(int argc,wchar_t** argv){try{
    require(argc==2,"rife_cancellation_tests runtime");
    Device dev;auto a=dev.upload(pattern(1920,1080,0,true),1920,1080,0,true);
    auto b=dev.upload(pattern(1920,1080,8,true),1920,1080,400000,true);
    std::vector<int64_t> times;for(int i=1;i<=32;++i)times.push_back(i*400000/33);
    auto e=engine(dev,argv[1]);std::vector<std::string> reference;
    {auto r=e->interpolate_pair(a,b,times);for(auto& f:r.frames)reference.push_back(hash_bytes(dev.read(f)));}
    for(bool inFlight:{false,true}){
        std::atomic<bool> cancelled{!inFlight};
        const auto before=MotionSynthesizer::inference_count();
        if(inFlight)e->set_wait_service([&]{
            if(MotionSynthesizer::inference_count()>before)cancelled.store(true,std::memory_order_release);
            return HANDLE(nullptr);
        });
        bool caught=false;try{e->interpolate_pair(a,b,times,&cancelled);}catch(const GpuWorkCancelled&){caught=true;}
        e->set_wait_service({});
        const auto submitted=MotionSynthesizer::inference_count()-before;
        require(caught,"cancellation did not reach caller");
        require(inFlight?(submitted>0&&submitted<32):submitted==0,"cancelled pair kept submitting phases");
        e->reset();
        auto next=e->interpolate_pair(a,b,times);
        require(next.frames.size()==reference.size(),"resume output count changed");
        for(size_t i=0;i<next.frames.size();++i)require(hash_bytes(dev.read(next.frames[i]))==reference[i],"resume reused cancelled pixels");
        std::cout<<"PASS inFlight="<<inFlight<<" submitted="<<submitted<<" exactResume="<<next.frames.size()<<'\n';
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
