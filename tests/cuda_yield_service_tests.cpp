#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include <d3d11_4.h>
int wmain(int argc,wchar_t**argv){try{
 require(argc==2,"runtime required");Device dev;auto e=engine(dev,argv[1]);
 auto a=pattern(1920,1080,0,false),b=pattern(1920,1080,8,false);auto fa=dev.upload(a,1920,1080,0),fb=dev.upload(b,1920,1080,400000);
 ComPtr<ID3D11Device5>d5;ComPtr<ID3D11DeviceContext4>c4;check(dev.d.As(&d5),"device5");check(dev.c.As(&c4),"context4");ComPtr<ID3D11Fence> renderer;check(d5->CreateFence(0,D3D11_FENCE_FLAG_NONE,IID_PPV_ARGS(&renderer)),"renderer marker fence");
 HANDLE returned=CreateEventW(nullptr,FALSE,FALSE,nullptr);require(returned!=nullptr,"renderer event");
 unsigned callbacks=0,early=0,marked=0;UINT64 value=0;
 for(int k=0;k<32;++k){
  const auto before=MotionSynthesizer::inference_count();bool submitted=false,completed=false;
  e->set_wait_service([&]()->HANDLE{
   // During the scene wait inference has not been submitted. Exercise only
   // the CUDA-ready wait, exactly as a renderer services earlier outputs.
   if(MotionSynthesizer::inference_count()<before+3)return nullptr;
   ++callbacks;
   if(!submitted){submitted=true;++marked;check(c4->Signal(renderer.Get(),++value),"renderer marker");dev.c->Flush();check(renderer->SetEventOnCompletion(value,returned),"renderer wake");}
   else if(renderer->GetCompletedValue()>=value){if(!completed)++early;completed=true;return nullptr;}
   return returned;
  });
  auto result=e->interpolate_pair(fa,fb,{100000,200000,300000});dev.wait();require(result.frames.size()==3,"missing outputs");
  // A completion-priority return can leave one harmless renderer hint set.
  WaitForSingleObject(returned,0);
 }
 e->set_wait_service({});CloseHandle(returned);
 require(callbacks>0&&marked>0,"No servicing during CUDA wait");
 require(early>0,"Shared D3D renderer made no observable progress during CUDA wait");
 require(dev.read(fa)==a&&dev.read(fb)==b,"Source modified");
 std::cout<<"PASS CUDA wait callbacks="<<callbacks<<" renderer markers="<<marked<<" renderer completions before CUDA wait returns="<<early<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
