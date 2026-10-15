#define wmain unused_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "nvof/gpu_phase_pipeline.hpp"
int wmain(int argc,wchar_t**argv){try{
 require(argc==3,"rife_cut_boundary_tests runtime native-case-folder");Device dev;auto e=engine(dev,argv[1]);
 const std::filesystem::path folder=argv[2];unsigned held=0,originals=0,cuts=0;
 for(const auto* name:{L"cut-44",L"cut-107",L"cut-144",L"cut-191"}){
  const int w=3840,h=1604;const size_t bytes=size_t(w)*h*3;
  auto pair=load(folder/name/L"pair.raw",bytes*2);std::vector<uint8_t> a(pair.begin(),pair.begin()+bytes),b(pair.begin()+bytes,pair.end());
  for(int pass=0;pass<3;++pass){
   // Exercise opening/seek and a cut following more than 12 identical cells.
   e->reset();int index=0,cutIndex=pass==1?16:1;int64_t stop=-1;unsigned emitted=0;auto metadata=std::make_shared<HdrMetadata>();
   GpuPhasePipeline pipeline({120000,1001},[&](const GpuFrame& x,const GpuFrame& y,const std::vector<int64_t>& ts){
    auto calls=MotionSynthesizer::inference_count();auto result=e->interpolate_pair(x,y,ts);
    require(result.quality.scene_cut==(index==cutIndex),"Hard cut/static classification changed");
    require(MotionSynthesizer::inference_count()==calls,"Cut or identical pair invoked TensorRT");cuts+=result.quality.scene_cut;return result;
   },{24000,1001});
   auto emit=[&](const GpuOutputFrame& out){
    require(stop<0||out.frame.pts==stop,"Boundary timestamp gap/overlap");stop=out.stop;
    const bool original=(emitted%5)==0;
    require(dev.read(out.frame)==(index==cutIndex&&original?b:a),"Cut output blended/warped an original");
    require(out.frame.hdr==metadata&&out.frame.hdr_transfer==15,"HDR metadata changed at cut");
    if(original)++originals;else if(index==cutIndex)++held;
    if(std::wstring(name)==L"cut-107"&&pass==0&&index==cutIndex&&!original&&emitted%5==2){auto raw=dev.read(out.frame);std::ofstream file(folder/L"after-hold.raw",std::ios::binary);file.write(reinterpret_cast<const char*>(raw.data()),raw.size());}
    ++emitted;return true;
   };
   for(index=0;index<=cutIndex;++index){auto frame=dev.upload(index==cutIndex?b:a,w,h,int64_t(index)*10010000/24,true);frame.hdr=metadata;frame.hdr_transfer=15;pipeline.push(frame,index==0,emit);}
  }
 }
 require(cuts==12&&held==48,"Not every cut phase verified");
 std::cout<<"PASS cuts="<<cuts<<" exact_held_phases="<<held<<" exact_originals="<<originals<<" fresh/still/seek HDR timestamps inference_calls="<<MotionSynthesizer::inference_count()<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
