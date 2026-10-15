#define wmain unused_sequence_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"
#include "nvof/gpu_phase_pipeline.hpp"
#include <set>
int wmain(int argc,wchar_t**argv){try{
    require(argc==4,"rife_sequence_hash runtime source-frames output.csv");Device dev;auto e=engine(dev,argv[1]);require(e->startup_warmup()>=0,"Synthetic startup warmup");std::ofstream report(argv[3]);report<<"index,pts,stop,sha256\n";
    std::set<int> expected{24,129,201};int index=0,outputs=0,cuts=0;int64_t stop=-1;
    GpuPhasePipeline pipeline({120000,1001},[&](const GpuFrame&a,const GpuFrame&b,const std::vector<int64_t>& times){auto batch=e->interpolate_pair(a,b,times);require(batch.quality.scene_cut==bool(expected.count(index)),"Sequence cut decision changed");cuts+=batch.quality.scene_cut;return batch;},{24000,1001});
    auto emit=[&](const GpuOutputFrame& output){require(stop<0||output.frame.pts==stop,"Timestamp gap/overlap");stop=output.stop;report<<outputs++<<','<<output.frame.pts<<','<<output.stop<<','<<hash_bytes(dev.read(output.frame))<<'\n';return true;};
    for(index=0;index<240;++index){wchar_t name[32]{};swprintf_s(name,L"frame-%03d.nv12",index);auto bytes=load(std::filesystem::path(argv[2])/name,size_t(1920)*1080*3/2);auto frame=dev.upload(bytes,1920,1080,int64_t(index)*10010000/24);pipeline.push(frame,index==0,emit);}
    require(cuts==3&&outputs==1196,"Sequence output count changed");std::cout<<"PASS source=240 output="<<outputs<<" cuts="<<cuts<<" engine_loads="<<MotionSynthesizer::engine_load_count()<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
