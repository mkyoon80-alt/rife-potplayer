#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include <bcrypt.h>
#include <iomanip>
#include <sstream>
#include "rife_hash.hpp"
int wmain(int argc,wchar_t**argv){timeBeginPeriod(1);try{
    require(argc==4,"rife_quality_hash runtime fixture-folder output.csv");Device dev;auto e=engine(dev,argv[1]);require(e->startup_warmup()>=0,"Synthetic startup warmup");
    std::ifstream manifest(std::filesystem::path(argv[2])/L"manifest.txt");std::ofstream report(argv[3]);require(bool(manifest)&&bool(report),"Quality files missing");
    report<<"case,pass,pts,cut,mask,hash\n";std::string id;int w,h,ten,transfer,cut;unsigned count=0;
    while(manifest>>id>>w>>h>>ten>>transfer>>cut){
        size_t n=size_t(w)*h*3/2*(ten?2:1);auto raw=load(std::filesystem::path(argv[2])/id/L"pair.raw",n*2);
        std::vector<uint8_t>a(raw.begin(),raw.begin()+n),b(raw.begin()+n,raw.end());auto fa=dev.upload(a,w,h,0,ten,transfer),fb=dev.upload(b,w,h,400000,ten,transfer);
        auto meta=std::make_shared<HdrMetadata>();meta->data[0].assign(80,37);fa.hdr=meta;
        const std::vector<int64_t> pts{50000,80000,100000,133337,160000,200000,240000,300000,320000,350000};
        // Cold first call, warm replay, reset/seek, then a fresh context/file.
        for(int pass=0;pass<4;++pass){
            if(pass==2)e->reset();if(pass==3){e.reset();e=engine(dev,argv[1]);require(e->startup_warmup()>=0,"Fresh synthetic startup warmup");}
            auto output=e->interpolate_pair(fa,fb,pts);dev.wait();
            if(cut>=0)require(output.quality.scene_cut==bool(cut),"Scene decision changed");
            if(output.quality.scene_cut){require(output.frames.empty(),"Cut must not warp");report<<id<<','<<pass<<",0,1,0,"<<hash_bytes(a)<<'\n';continue;}
            require(output.frames.size()==pts.size(),"Missing phase");
            for(auto& f:output.frames){require(f.hdr==meta&&f.hdr_transfer==unsigned(transfer),"HDR metadata changed");auto bytes=dev.read(f);report<<id<<','<<pass<<','<<f.pts<<",0,"<<output.quality.native_synthesized_mask<<','<<hash_bytes(bytes)<<'\n';++count;}
        }
        require(dev.read(fa)==a&&dev.read(fb)==b,"Source bytes changed");report.flush();std::cout<<id<<" quality complete\n"<<std::flush;
    }
    require(MotionSynthesizer::engine_load_count()==1,"Unexpected engine reload");std::cout<<"PASS output_hashes="<<count<<" engine_loads=1\n";std::cout<<"GRAPH launches="<<MotionSynthesizer::graph_launch_count()<<" builds="<<MotionSynthesizer::graph_build_count()<<" fallbacks="<<MotionSynthesizer::graph_fallback_count()<<std::endl;timeEndPeriod(1);return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';timeEndPeriod(1);return 1;}}
