#define wmain unused_head_cache_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"
int wmain(int argc,wchar_t** argv){try {
    require(argc==2,"rife_head_cache_tests runtime");Device dev;
    SetEnvironmentVariableW(L"RIFE_EXPERIMENT_FINAL_MOTION_COPY",nullptr);
    SetEnvironmentVariableW(L"RIFE_HEAD_CACHE_SYNC_PREPARE",L"1");
    std::vector<std::string> reference;size_t compared=0,pairCount=0,phaseCount=0;
    for(bool cached:{false,true}){
        SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",cached?L"1":L"0");
#ifdef RIFE_TEST_PACKED_COMBINED
        SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PACKED_MOTION",cached?L"1":L"0");
#endif
        auto e=engine(dev,argv[1]);size_t index=0;
        std::vector<GpuFrame> retained;std::vector<std::string> hashes;
        auto test=[&](const GpuFrame& a,const GpuFrame& b,int n){
            std::vector<int64_t> times;for(int k=1;k<=n;++k)times.push_back(a.pts+int64_t(k)*(b.pts-a.pts)/(n+1));
            if(n>=3){times.front()=a.pts+1;times.back()=b.pts-1;}
            const auto hp=MotionSynthesizer::head_cache_pair_count(),hs=MotionSynthesizer::head_cache_phase_count();
            auto batch=e->interpolate_pair(a,b,times);dev.wait();
            require(batch.frames.size()==times.size(),"missing phases");
            require(MotionSynthesizer::head_cache_pair_count()-hp==uint64_t(cached&&n>1),"Head must run exactly once per multi-output pair, never single output");
            require(MotionSynthesizer::head_cache_phase_count()-hs==uint64_t(cached&&n>1?n:0),"incorrect selected phase path");
            if(cached){pairCount+=n>1;phaseCount+=n>1?n:0;}
            for(size_t k=0;k<times.size();++k){
                auto& f=batch.frames[k];require(f.pts==times[k]&&f.hdr==a.hdr&&f.hdr_transfer==a.hdr_transfer,"PTS/HDR mismatch");
                const auto hash=hash_bytes(dev.read(f));
                if(cached){require(hash==reference.at(index),"split engine changed final Y/UV pixels");++compared;}else reference.push_back(hash);
                ++index;
                if(a.width==320&&(n==4||n==5||n==9||n==32)){retained.push_back(f);hashes.push_back(hash);}
            }
        };
        for(auto spec:std::vector<std::array<int,4>>{{320,180,0,0},{1920,1080,1,0},{3840,2160,1,15},{7680,4320,1,16},{360,640,1,0},{2560,1080,0,0}}){
            int w=spec[0],h=spec[1];auto a=pattern(w,h,0,spec[2]),b=pattern(w,h,8,spec[2]);
            auto fa=dev.upload(a,w,h,0,spec[2],spec[3]),fb=dev.upload(b,w,h,400003,spec[2],spec[3]);
            auto meta=std::make_shared<HdrMetadata>();meta->data[0].assign(80,37);meta->data[1].assign(8,19);fa.hdr=meta;
            std::vector<int> counts=w==320?std::vector<int>{}:std::vector<int>{1,2,3,4,5,8,9};
            if(w==320)for(int n=1;n<=32;++n)counts.push_back(n);
            if(w==7680)counts={1,2,3,4,5};
            for(int n:counts){if(n%5==0)e->reset();test(fa,fb,n);}
            require(dev.read(fa)==a&&dev.read(fb)==b,"source pixels changed");
            if(w==320){
                for(int serial=0;serial<6;++serial){
                    // Same PTS and same physical textures, different source bytes.
                    dev.wait();a=pattern(w,h,float(serial*3+1),false);b=pattern(w,h,float(serial*3+9),false);
                    dev.c->UpdateSubresource(fa.texture.Get(),0,nullptr,a.data(),w,0);dev.c->UpdateSubresource(fb.texture.Get(),0,nullptr,b.data(),w,0);
                    e->reset();test(fa,fb,serial%2?1:5);
                }
            }
        }
        e.reset();dev.wait();for(size_t i=0;i<retained.size();++i)require(hash_bytes(dev.read(retained[i]))==hashes[i],"output changed after feature reuse/teardown");
        if(cached)require(index==reference.size(),"incomplete parity test");
    }
    require(MotionSynthesizer::head_cache_load_count()==1,"split plans repeatedly loaded");
    std::cout<<"PASS exactOutputs="<<compared<<" cachedPairs="<<pairCount<<" cachedPhases="<<phaseCount
        <<" planLoads="<<MotionSynthesizer::head_cache_load_count()<<" contextPairs="<<MotionSynthesizer::head_cache_context_count()
        <<" phases=1..32 single-bypass/P010/SDR/PQ/HLG/8K/portrait/ultrawide/VFR/retained/same-PTS-same-textures/seek\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
