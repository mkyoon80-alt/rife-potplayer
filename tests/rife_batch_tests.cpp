#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"
int wmain(int argc,wchar_t** argv){try{
    require(argc==2,"rife_batch_tests runtime");Device dev;std::vector<std::string> reference;
    for(bool batched:{false,true}){
        SetEnvironmentVariableW(L"RIFE_DISABLE_BATCH_HANDOFF",batched?nullptr:L"1");
        auto e=engine(dev,argv[1]);size_t index=0;
        for(auto spec:std::vector<std::array<int,4>>{{1920,1080,0,0},{3840,2160,1,15},{7680,4320,1,15},{1080,1920,0,0},{640,360,1,16}}){
            int w=spec[0],h=spec[1];auto a=pattern(w,h,0,spec[2]),b=pattern(w,h,8,spec[2]);
            auto fa=dev.upload(a,w,h,0,spec[2],spec[3]),fb=dev.upload(b,w,h,400000,spec[2],spec[3]);
            auto meta=std::make_shared<HdrMetadata>();meta->data[0].assign(80,37);fa.hdr=meta;
            std::vector<std::vector<int64_t>> rounds{{200000},{100000,200000,300000},{133337,266663},{100000,200000,300000}};
            if(w==640){rounds.push_back({1,80000,100000,133337,200000,240000,300000,320000,399999});std::vector<int64_t> many;for(int k=1;k<=32;++k)many.push_back(k*12000);rounds.push_back(many);}
            for(size_t r=0;r<rounds.size();++r){
                if(r==3)e->reset();if(r==2)e->enable_timing(true,false,true);
                auto before=MotionSynthesizer::handoff_count();auto output=e->interpolate_pair(fa,fb,rounds[r]);dev.wait();
                require(output.frames.size()==rounds[r].size(),"Missing batch output");
                const auto expected=batched?(rounds[r].size()+3)/4:rounds[r].size();
                require(MotionSynthesizer::handoff_count()-before==expected,"Unexpected ownership handoff count");
                for(size_t k=0;k<output.frames.size();++k){auto& frame=output.frames[k];
                    require(frame.pts==rounds[r][k]&&frame.hdr==meta&&frame.hdr_transfer==unsigned(spec[3]),"Batch changed timestamps/HDR metadata");
                    auto hash=hash_bytes(dev.read(frame));if(!batched)reference.push_back(hash);else require(hash==reference.at(index),"Batch changed output pixels");++index;
                }
            }
            require(dev.read(fa)==a&&dev.read(fb)==b,"Batch modified source pixels");
        }
        if(batched)require(index==reference.size(),"Missing comparison");
    }
    require(MotionSynthesizer::engine_load_count()==1,"Batch reloaded inference plan");
    std::cout<<"PASS batch handoffs 3->1, chunk limits, 32 phases, VFR, SDR/PQ/HLG/8K/portrait, seek, diagnostics, unchanged source and exact outputs="<<reference.size()<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
