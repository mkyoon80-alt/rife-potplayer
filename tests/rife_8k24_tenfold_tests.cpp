#define wmain unused_tenfold_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"

int wmain(int argc,wchar_t** argv){try{
    require(argc==2,"rife_8k24_tenfold_tests runtime");Device dev;
    SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",L"1");
    SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PACKED_MOTION",L"1");
    SetEnvironmentVariableW(L"RIFE_HEAD_CACHE_SYNC_PREPARE",L"1");
    size_t compared=0;
    for(auto spec:std::vector<std::array<int,3>>{{2560,1440,0},{7680,4320,16}}){
        const int w=spec[0],h=spec[1];const unsigned transfer=spec[2];
        auto a=pattern(w,h,0,true),b=pattern(w,h,8,true);
        auto fa=dev.upload(a,w,h,0,true,transfer),fb=dev.upload(b,w,h,416667,true,transfer);
        std::vector<int64_t> times;for(int k=1;k<10;++k)times.push_back(416667LL*k/10);
        std::vector<std::string> reference;
        for(bool candidate:{false,true}){
            SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",candidate?L"1":L"0");
            SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PACKED_MOTION",candidate?L"1":L"0");
            auto e=engine(dev,argv[1]);
            for(int serial=0;serial<2;++serial){
                if(serial){dev.wait();a=pattern(w,h,3,true);b=pattern(w,h,11,true);
                    dev.c->UpdateSubresource(fa.texture.Get(),0,nullptr,a.data(),w*2,0);
                    dev.c->UpdateSubresource(fb.texture.Get(),0,nullptr,b.data(),w*2,0);e->reset();}
                const auto hp=MotionSynthesizer::head_cache_pair_count();
                auto batch=e->interpolate_pair(fa,fb,times);require(batch.frames.size()==9,"8K24 x10 missing phases");
                require(MotionSynthesizer::head_cache_pair_count()-hp==unsigned(candidate),"real Head was reused from dummy/previous pixels");
                for(size_t k=0;k<times.size();++k){
                    require(batch.frames[k].pts==times[k]&&batch.frames[k].hdr_transfer==transfer,"PTS/transfer changed");
                    const auto hash=hash_bytes(dev.read(batch.frames[k]));
                    if(candidate){require(hash==reference.at(serial*9+k),"selected E03/E04 path changed final pixels");++compared;}
                    else reference.push_back(hash);
                }
            }
            // Restore the exact initial images for the other implementation.
            dev.wait();a=pattern(w,h,0,true);b=pattern(w,h,8,true);
            dev.c->UpdateSubresource(fa.texture.Get(),0,nullptr,a.data(),w*2,0);
            dev.c->UpdateSubresource(fb.texture.Get(),0,nullptr,b.data(),w*2,0);
        }
    }
    std::cout<<"PASS exactOutputs="<<compared<<" QHD/8K24x10 reference-versus-combined unchanged-PTS/transfer real-Head-per-pair seek/same-texture/same-PTS\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
