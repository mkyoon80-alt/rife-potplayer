#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
int wmain(int argc,wchar_t**argv){timeBeginPeriod(1);try{require(argc==2,"runtime path");Device dev;auto e=engine(dev,argv[1]);
 for(auto spec:std::vector<std::array<int,4>>{{1920,1080,0,0},{3840,2160,1,15},{3840,2160,1,16},{7680,4320,0,0}}){
  int w=spec[0],h=spec[1];std::vector<GpuFrame> f;for(int i=0;i<5;++i)f.push_back(dev.upload(pattern(w,h,float(i)*8,spec[2]),w,h,int64_t(i)*400000,spec[2],spec[3]));
  for(int phases:{1,3}){std::vector<double> samples;for(int i=0;i<160;++i){int k=i%4;auto a=f[k],b=f[k+1];a.pts=int64_t(i)*400000;b.pts=a.pts+400000;
    std::vector<int64_t> pts=phases==1?std::vector<int64_t>{a.pts+200000}:std::vector<int64_t>{a.pts+100000,a.pts+200000,a.pts+300000};
    auto start=Clock::now();auto batch=e->interpolate_pair(a,b,pts);dev.wait();double ms=elapsed(start);require(batch.frames.size()==phases&&!batch.quality.scene_cut,"Motion benchmark rejected");if(i>=40)samples.push_back(ms);
   }std::sort(samples.begin(),samples.end());std::cout<<"BENCH "<<w<<'x'<<h<<" transfer="<<spec[3]<<" phases="<<phases<<" median_ms="<<samples[60]<<" mean_ms="<<std::accumulate(samples.begin(),samples.end(),0.)/samples.size()<<" p95_ms="<<samples[114]<<std::endl;
  }
  double maximum=0.;for(int j=0;j<30;++j){auto start=Clock::now();e->reset();maximum=(std::max)(maximum,elapsed(start));}std::cout<<"RESET "<<w<<'x'<<h<<" max_ms="<<maximum<<std::endl;
 }
 require(MotionSynthesizer::engine_load_count()==1,"Engine reloaded");std::cout<<"PASS engine_loads=1\n";timeEndPeriod(1);return 0;
 }catch(const std::exception&e){std::cerr<<e.what();timeEndPeriod(1);return 1;}}
