#define wmain unused_async_head_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "rife_hash.hpp"
int wmain(int argc,wchar_t**argv){try{
 require(argc==3,"runtime ready|missing");const bool missing=std::wstring(argv[2])==L"missing";
 SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",L"0");SetEnvironmentVariableW(L"RIFE_HEAD_CACHE_SYNC_PREPARE",nullptr);
 Device dev;auto a=dev.upload(pattern(1920,1080,0,true),1920,1080,0,true),b=dev.upload(pattern(1920,1080,8,true),1920,1080,400000,true);
 std::vector<int64_t> times{80000,160000,240000,320000};std::vector<std::string> ref;
 {auto e=engine(dev,argv[1]);auto r=e->interpolate_pair(a,b,times);for(auto& f:r.frames)ref.push_back(hash_bytes(dev.read(f)));}
 SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",L"1");
 {auto e=engine(dev,argv[1]);for(int i=0;i<4;++i){e->interpolate_pair(a,b,{200000});dev.wait();}require(MotionSynthesizer::head_cache_load_count()==0,"single output loaded split plans");}
 if(missing){
  auto e=engine(dev,argv[1]);const auto began=Clock::now();
  while(!e->startup_ready()){
   e->startup_warmup(1920,1080,true);Sleep(1);
   require(elapsed(began)<15000,"optional missing-plan startup did not fall back");
  }
  auto r=e->interpolate_pair(a,b,times);
  for(size_t i=0;i<r.frames.size();++i)require(hash_bytes(dev.read(r.frames[i]))==ref.at(i),"startup fallback changed pixels");
 }
 auto start=Clock::now();unsigned loops=0,compared=0;
 do{
  auto e=engine(dev,argv[1]);auto r=e->interpolate_pair(a,b,times);e->reset();e.reset();
  for(size_t i=0;i<r.frames.size();++i){require(hash_bytes(dev.read(r.frames[i]))==ref.at(i),"asynchronous preparation/reset changed pixels");++compared;}
  ++loops;Sleep(10);require(elapsed(start)<15000,"asynchronous preparation timed out");
 }while(loops<32||(!missing&&MotionSynthesizer::head_cache_pair_count()==0));
 require(missing?MotionSynthesizer::head_cache_load_count()==0:MotionSynthesizer::head_cache_load_count()==1,"unexpected split plan count");
 require(missing?MotionSynthesizer::head_cache_pair_count()==0:MotionSynthesizer::head_cache_pair_count()>0,"wrong fallback/cache path");
 std::cout<<"PASS async cycles="<<loops<<" exactOutputs="<<compared<<" cachePairs="<<MotionSynthesizer::head_cache_pair_count()<<" planLoads="<<MotionSynthesizer::head_cache_load_count()<<" missing="<<missing<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
