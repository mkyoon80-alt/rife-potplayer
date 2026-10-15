#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"
int wmain(int argc,wchar_t** argv){try{
    require(argc==2,"rife_source_views_tests runtime");Device dev;auto e=engine(dev,argv[1]);
    unsigned compared=0;std::vector<GpuFrame> retained;std::vector<std::string> hashes;
    for(auto spec:std::vector<std::array<int,4>>{{640,360,0,0},{1280,720,1,15},{360,640,0,0},{640,360,1,16},{640,360,0,0}}){
        const int w=spec[0],h=spec[1];
        auto a=pattern(w,h,0,spec[2]),b=pattern(w,h,8,spec[2]);
        auto fa=dev.upload(a,w,h,0,spec[2],spec[3]),fb=dev.upload(b,w,h,400000,spec[2],spec[3]);
        auto meta=std::make_shared<HdrMetadata>();meta->data[0].assign(80,37);fa.hdr=meta;
        auto fallback=e->interpolate_pair(fa,fb,{100000,200000,300000});
        std::vector<std::string> reference;for(auto& frame:fallback.frames)reference.push_back(hash_bytes(dev.read(frame)));
        for(int round=0;round<4;++round){
            e->reset();auto ownedA=e->copy(fa),ownedB=e->copy(fb);
            auto output=e->interpolate_pair(ownedA,ownedB,{100000,200000,300000});
            require(output.frames.size()==3,"pooled source did not synthesize all phases");
            for(size_t i=0;i<3;++i){
                auto hash=hash_bytes(dev.read(output.frames[i]));require(hash==reference[i],"reused source views changed pixels");
                require(output.frames[i].hdr==meta&&output.frames[i].hdr_transfer==unsigned(spec[3]),"source-view reuse changed metadata");
                retained.push_back(output.frames[i]);hashes.push_back(hash);++compared;
            }
        }
    }
    // More than the pool cap may stay leased downstream. Overflow surfaces
    // must fall back safely; no output already held by the renderer may change.
    auto a=dev.upload(pattern(640,360,0,false),640,360,0),b=dev.upload(pattern(640,360,8,false),640,360,400000);
    for(int i=0;i<10;++i){auto aa=e->copy(a),bb=e->copy(b);auto output=e->interpolate_pair(aa,bb,{200000});retained.push_back(output.frames[0]);hashes.push_back(hash_bytes(dev.read(output.frames[0])));}
    e->reset();for(size_t i=0;i<retained.size();++i)require(hash_bytes(dev.read(retained[i]))==hashes[i],"leased output reused or stale view retained");
    std::cout<<"PASS pooled/fallback source views: "<<compared<<" exact outputs; "<<retained.size()<<" retained outputs; SDR/HDR10/HLG/portrait/reset/pool overflow\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
