#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "nvof/scene_cut.hpp"
int wmain(int argc,wchar_t** argv){try{
 require(argc==3,"cut_native_audit runtime native-folder");Device dev;auto e=engine(dev,argv[1]);
 const std::filesystem::path folder=argv[2];std::ifstream manifest(folder/L"manifest.txt");require(bool(manifest),"Manifest missing");
 std::ofstream csv(folder/L"results.csv");csv<<"id,width,height,ten_bit,transfer,expected_cut,actual_cut,cpu_cut,difference,coverage,histogram,variation,inference_calls,frames,ms\n";
 std::string id;int w,h,ten,transfer,expected,total=0,fp=0,fn=0;
 while(manifest>>id>>w>>h>>ten>>transfer>>expected){
  const size_t size=size_t(w)*h*3/2*(ten?2:1);auto both=load(folder/id/L"pair.raw",size*2);
  std::vector<uint8_t>a(both.begin(),both.begin()+size),b(both.begin()+size,both.end()),ya(size_t(w)*h),yb(ya.size());
  for(size_t i=0;i<ya.size();++i){ya[i]=ten?uint8_t(std::lround(reinterpret_cast<uint16_t*>(a.data())[i]*255.0/65535)):a[i];yb[i]=ten?uint8_t(std::lround(reinterpret_cast<uint16_t*>(b.data())[i]*255.0/65535)):b[i];}
  SceneDetector fresh;auto decision=fresh.evaluate(scene_stats_cpu(ya.data(),yb.data(),w,h));e->reset();
  auto fa=dev.upload(a,w,h,0,ten,transfer),fb=dev.upload(b,w,h,400000,ten,transfer);
  auto meta=std::make_shared<HdrMetadata>();meta->data[0].assign(80,37);fa.hdr=meta;
  const auto calls=MotionSynthesizer::inference_count();auto start=Clock::now();auto batch=e->interpolate_pair(fa,fb,{100000,200000,300000});dev.wait();auto ms=elapsed(start);auto inferences=MotionSynthesizer::inference_count()-calls;
  require(batch.quality.scene_cut==decision.cut,"GPU vs exact CPU classification mismatch");
  if(batch.quality.scene_cut)require(batch.frames.empty()&&inferences==0,"Cut did not bypass RIFE");
  else{require(batch.frames.size()==3,"Unclassified pair missing outputs");for(auto& f:batch.frames)require(f.hdr==meta&&f.hdr_transfer==unsigned(transfer),"Metadata not preserved");auto midpoint=dev.read(batch.frames[1]);std::ofstream raw(folder/id/L"midpoint.raw",std::ios::binary);raw.write(reinterpret_cast<const char*>(midpoint.data()),midpoint.size());}
  fp+=decision.cut&&!expected;fn+=!decision.cut&&expected;++total;
  csv<<id<<','<<w<<','<<h<<','<<ten<<','<<transfer<<','<<expected<<','<<decision.cut<<','<<decision.cut<<','<<decision.difference<<','<<decision.coverage<<','<<decision.histogram_distance<<','<<decision.variation<<','<<inferences<<','<<batch.frames.size()<<','<<ms<<'\n';csv.flush();
  std::cout<<id<<" expected="<<expected<<" actual="<<decision.cut<<" inferences="<<inferences<<" frames="<<batch.frames.size()<<std::endl;
 }
 require(MotionSynthesizer::engine_load_count()==(MotionSynthesizer::inference_count()?1u:0u),"Unexpected engine loading across cases");
 std::cout<<"COMPLETE cases="<<total<<" false_positive="<<fp<<" missed_cut="<<fn<<" loads="<<MotionSynthesizer::engine_load_count()<<std::endl;
 require(fp==0&&fn==0,"Native source scene classification failed");return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
