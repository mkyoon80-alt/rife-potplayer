#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
int wmain(int argc,wchar_t**argv){timeBeginPeriod(1);try{require(argc==3,"runtime source-native-folder");Device dev;auto e=engine(dev,argv[1]);const std::filesystem::path folder=argv[2];
 struct Case{const char* name;int w,h,ten;};for(auto spec:std::vector<Case>{{"v02_w0_f14",1920,1080,0},{"v13_w2_f46",3840,1604,1},{"v10_w0_f33",1280,720,1},{"v11_w0_f99",1920,1080,1}}){
  size_t size=size_t(spec.w)*spec.h*3/2*(spec.ten?2:1);auto bytes=load(folder/spec.name/L"pair.raw",size*2);std::vector<uint8_t>a(bytes.begin(),bytes.begin()+size),b(bytes.begin()+size,bytes.end());auto fa=dev.upload(a,spec.w,spec.h,0,spec.ten),fb=dev.upload(b,spec.w,spec.h,400000,spec.ten);
  for(int phases:{1,3}){std::vector<double> samples;auto pts=phases==1?std::vector<int64_t>{200000}:std::vector<int64_t>{100000,200000,300000};
   for(int i=0;i<160;++i){auto start=Clock::now();auto batch=e->interpolate_pair(fa,fb,pts);dev.wait();double ms=elapsed(start);require(batch.frames.size()==phases&&!batch.quality.scene_cut,"Actual motion was rejected");if(i>=40)samples.push_back(ms);}
   std::sort(samples.begin(),samples.end());std::cout<<"BENCH "<<spec.name<<" phases="<<phases<<" median_ms="<<samples[60]<<" mean_ms="<<std::accumulate(samples.begin(),samples.end(),0.)/samples.size()<<" p95_ms="<<samples[114]<<std::endl;
  }
 }require(MotionSynthesizer::engine_load_count()==1,"Plan reloaded");std::cout<<"PASS engine_loads=1\n";timeEndPeriod(1);return 0;
 }catch(const std::exception&e){std::cerr<<e.what();timeEndPeriod(1);return 1;}}
