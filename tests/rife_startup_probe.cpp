#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
int wmain(int argc,wchar_t**argv){timeBeginPeriod(1);try{
 require(argc>=2,"runtime required");Device dev;

 auto a=pattern(7680,4320,0,true),b=pattern(7680,4320,8,true);
 auto fa=dev.upload(a,7680,4320,0,true,15),fb=dev.upload(b,7680,4320,166667,true,15);dev.wait();
 if(argc>2){auto t=Clock::now();require(MotionSynthesizer::preload_async(argv[1],dev.d.Get()),"preload request");std::cout<<"PRELOAD requestMs="<<elapsed(t)<<std::endl;if(std::wstring(argv[2])==L"ready")Sleep(750);}
 for(int file=0;file<3;++file){
  auto e=engine(dev,argv[1]);e->enable_timing(true,true,1);
  auto start=Clock::now();auto original=e->copy(fa,true);dev.wait();double originalMs=elapsed(start);
  for(int pair=0;pair<3;++pair){
   start=Clock::now();auto result=e->interpolate_pair(fa,fb,{41667,83333,125000});dev.wait();double total=elapsed(start);auto t=e->pair_timing();require(t.rife!=nullptr,"timing required");auto r=*t.rife;
   double enq=0;for(unsigned i=0;i<r.phase_count;++i)enq+=r.phases[i].enqueue_cpu_ms;
   std::cout<<"STARTUP file="<<file<<" pair="<<pair<<" originalMs="<<originalMs<<" totalMs="<<total<<" prepareMs="<<t.prepare_ms<<" initMs="<<r.initialize_cpu_ms<<" engineMs="<<r.init_engine_ms<<" contextMs="<<r.init_context_ms<<" buffersMs="<<r.init_buffers_ms<<" shadersMs="<<r.init_shaders_ms<<" enqueueMs="<<enq<<" sceneWaitMs="<<t.wait_each_ms[0]<<" cudaWaitMs="<<t.cuda_ready_wait_ms<<" loads="<<MotionSynthesizer::engine_load_count()<<" contexts="<<MotionSynthesizer::execution_create_count()<<'\n';
  }
 }
 timeEndPeriod(1);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';timeEndPeriod(1);return 1;}}
