#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"
int wmain(int argc,wchar_t** argv){try {
    SetEnvironmentVariableW(L"RIFE_EXPERIMENT_FINAL_MOTION_COPY",L"1");
    require(argc==2||argc==3,"rife_final_copy_tests runtime [hashes.txt]");Device dev;
    std::vector<std::string> reference;size_t compared=0,copiesBefore=0,copiesAfter=0;
    for(bool elide:{false,true}) {
        SetEnvironmentVariableW(L"RIFE_REFERENCE_FINAL_MOTION_COPY",elide?nullptr:L"1");
        auto e=engine(dev,argv[1]);size_t index=0;
        std::vector<GpuFrame> retained;std::vector<std::string> retainedHashes;
        for(auto spec:std::vector<std::array<int,4>>{{320,180,0,0},{1920,1080,1,0},{3840,2160,1,15},{7680,4320,1,16},{360,640,1,0},{2560,1080,0,0}}) {
            int w=spec[0],h=spec[1];auto a=pattern(w,h,0,spec[2]),b=pattern(w,h,8,spec[2]);
            auto fa=dev.upload(a,w,h,0,spec[2],spec[3]),fb=dev.upload(b,w,h,400003,spec[2],spec[3]);
            auto meta=std::make_shared<HdrMetadata>();meta->data[0].assign(80,37);meta->data[1].assign(8,19);fa.hdr=meta;
            std::vector<int> counts=w==320?std::vector<int>{}:std::vector<int>{1,4,5,8,9};
            if(w==320)for(int n=1;n<=32;++n)counts.push_back(n);
            if(w==7680)counts={3,4,5};
            for(int n:counts) {
                if(n%5==0)e->reset();e->enable_timing(true,true);
                std::vector<int64_t> times;for(int k=1;k<=n;++k)times.push_back(int64_t(k)*400003/(n+1));
                if(n>=3){times.front()=1;times.back()=400002;}
                auto before=MotionSynthesizer::handoff_count();auto output=e->interpolate_pair(fa,fb,times);dev.wait();
                require(output.frames.size()==times.size(),"missing output");
                require(MotionSynthesizer::handoff_count()-before==uint64_t((n+3)/4),"ownership handoff count changed");
                auto timing=e->pair_timing().rife;require(timing&&timing->phase_count==unsigned(n),"phase diagnostic missing");
                unsigned copies=0;for(unsigned j=0;j<timing->phase_count;++j){auto& t=timing->phases[j];copies+=t.copy_count;if(!t.copy_count)require(t.copy_cpu_ms==0,"skipped copy reported CPU copy time");}
                require(copies==unsigned(n==1?0:elide?n-(n+3)/4:n),"incorrect physical copy count");
                if(elide)copiesAfter+=copies;else copiesBefore+=copies;
                for(size_t k=0;k<times.size();++k) {
                    auto& f=output.frames[k];require(f.pts==times[k]&&f.hdr==meta&&f.hdr_transfer==unsigned(spec[3]),"PTS/HDR changed");
                    auto hash=hash_bytes(dev.read(f));
                    if(!elide)reference.push_back(hash);else {require(hash==reference.at(index),"scratch path changed output pixels");++compared;}
                    ++index;
                    if(w==320 && (n==4||n==5||n==9||n==32)){retained.push_back(f);retainedHashes.push_back(hash);}
                }
            }
            require(dev.read(fa)==a&&dev.read(fb)==b,"source pixels changed");
        }
        e.reset();dev.wait();
        for(size_t i=0;i<retained.size();++i)require(hash_bytes(dev.read(retained[i]))==retainedHashes[i],"retained renderer output changed after scratch reuse/teardown");
        if(elide)require(index==reference.size(),"incomplete comparison");
    }
    if(argc==3){std::ofstream hashes(argv[2]);require(bool(hashes),"hash report");for(auto& value:reference)hashes<<value<<'\n';}
    std::cout<<"PASS exactOutputs="<<compared<<" copiesBefore="<<copiesBefore<<" copiesAfter="<<copiesAfter
             <<" phases=1..32 SDR/PQ/HLG/8K/portrait/ultrawide/VFR/near-endpoints/reset/retained-output\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
