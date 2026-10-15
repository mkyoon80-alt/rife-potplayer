#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "nvof/gpu_phase_pipeline.hpp"
#include <set>
int wmain(int argc,wchar_t** argv){try{
 require(argc==3,"rife_scene_regression runtime frame-folder");Device dev;auto e=engine(dev,argv[1]);
 const std::filesystem::path folder=argv[2];const int w=1920,h=1080;const size_t bytes=size_t(w)*h*3/2;
 auto path=[&](int n){wchar_t name[64]{};swprintf_s(name,L"frame-%03d.nv12",n);return folder/name;};
 std::set<int> expected{24,129,201};int index=0,cuts=0,output=0,held=0,originals=0;int64_t stop=-1;
 std::vector<uint8_t> current,previous;std::vector<double> cut_ms,motion_ms;
 GpuPhasePipeline pipeline({120000,1001},[&](const GpuFrame&a,const GpuFrame&b,const std::vector<int64_t>& times){
  const auto calls=MotionSynthesizer::inference_count();auto start=Clock::now();auto batch=e->interpolate_pair(a,b,times);dev.wait();double ms=elapsed(start);
  require(batch.quality.scene_cut==bool(expected.count(index)),"GPU scene decision differs from measured source boundary");
  if(batch.quality.scene_cut){++cuts;cut_ms.push_back(ms);require(batch.frames.empty()&&MotionSynthesizer::inference_count()==calls,"Cut must bypass TensorRT and return no warped frames");}
  else if(index>1)motion_ms.push_back(ms);
  return batch;
 },{24000,1001});
 auto emit=[&](const GpuOutputFrame& o){
  require(stop<0||o.frame.pts==stop,"Scene cut introduced timestamp gap/overlap");stop=o.stop;
  const bool original=output%5==0;
  if(original||expected.count(index)){
   auto raw=dev.read(o.frame);
   require(raw==(original?current:previous),"Original or boundary hold pixels changed");
   if(original)++originals;else ++held;
   if(index==129&&!original&&output%5==2){std::ofstream f(folder/L"after.nv12",std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),raw.size());}
  }
  ++output;return true;
 };
 for(index=0;index<240;++index){current=load(path(index),bytes);auto f=dev.upload(current,w,h,int64_t(index)*10010000/24);pipeline.push(f,index==0,emit);previous=current;}
 require(cuts==3&&held==12&&originals==240,"Not every cut phase/original was verified");
 require(MotionSynthesizer::engine_load_count()==1,"Scene cut reloaded TensorRT engine");
 auto mean=[](const std::vector<double>& v){return std::accumulate(v.begin(),v.end(),0.0)/v.size();};
 std::cout<<"PASS source=240 output="<<output<<" cuts="<<cuts<<" exact_held_phases="<<held<<" exact_originals="<<originals<<" engine_loads=1 cut_mean_ms="<<mean(cut_ms)<<" continuous_x5_mean_ms="<<mean(motion_ms)<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
