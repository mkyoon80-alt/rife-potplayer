#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"
int wmain(int argc,wchar_t** argv){try{
    require(argc==2,"rife_interop_tests runtime");Device dev;
    std::vector<std::string> reference;
    for(bool optimized:{false,true}){
        SetEnvironmentVariableW(L"RIFE_REFERENCE_INTEROP",optimized?nullptr:L"1");
        SetEnvironmentVariableW(L"RIFE_REFERENCE_COMPOSITE",optimized?nullptr:L"1");
        auto e=engine(dev,argv[1]);size_t index=0;
        for(auto spec:std::vector<std::array<int,4>>{{1920,1080,0,0},{3840,2160,1,15},{1080,1920,0,0},{640,360,1,16}}){
            int w=spec[0],h=spec[1];
            for(int pair=0;pair<3;++pair){
                auto a=pattern(w,h,float(pair*9),spec[2]),b=pattern(w,h,float(pair*9+8),spec[2]);
                auto fa=dev.upload(a,w,h,0,spec[2],spec[3]),fb=dev.upload(b,w,h,400000,spec[2],spec[3]);
                const std::vector<std::vector<int64_t>> times{{200000},{200000,300000},{100000,200000,300000},{133337},{200000}};
                for(size_t round=0;round<times.size();++round){
                    if(round==3)e->reset();
                    auto output=e->interpolate_pair(fa,fb,times[round]);dev.wait();
                    require(output.frames.size()==times[round].size()&&output.quality.native_synthesized_mask==((1u<<times[round].size())-1),"Missing interpolated output");
                    for(auto& frame:output.frames){
                        auto hash=hash_bytes(dev.read(frame));
                        if(!optimized)reference.push_back(hash);else require(hash==reference.at(index),"Midpoint cache / native plane / source update changed pixels");
                        ++index;
                        require(frame.hdr_transfer==unsigned(spec[3]),"Transfer metadata changed");
                    }
                }
                require(dev.read(fa)==a&&dev.read(fb)==b,"Read-only input was modified");
            }
        }
        if(optimized)require(index==reference.size(),"Missing comparison");
    }
    require(MotionSynthesizer::engine_load_count()==1,"File switch reloaded engine");
    std::cout<<"PASS interop phase cache, .5 first, fractional phases, changed sources, seek, portrait, HDR: "<<reference.size()<<" exact outputs\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
