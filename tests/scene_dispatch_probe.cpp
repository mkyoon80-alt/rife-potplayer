#define main unused_scene_identity_main
#include "scene_identity_tests.cpp"
#undef main
#include <d3dcompiler.h>
int wmain(int argc,wchar_t**argv){try{
 require(argc==3,"shader.hlsl output.csv");std::ifstream f(argv[1]);std::string source((std::istreambuf_iterator<char>(f)),{});require(!source.empty(),"shader source missing");
 Device dev;CheckScene scene(dev);std::array<ComPtr<ID3D11ComputeShader>,4> variants;const unsigned groups[]={64,256,1024,4096};
 for(int v=0;v<4;++v){auto s=source;const auto replacement=std::to_string(groups[v]*256);size_t pos=0;unsigned replaced=0;
  while((pos=s.find("1048576",pos))!=std::string::npos){s.replace(pos,7,replacement);pos+=replacement.size();++replaced;}require(replaced==4,"stride contract changed");
  ComPtr<ID3DBlob> code,error;const auto hr=D3DCompile(s.data(),s.size(),"scene-dispatch-experiment",nullptr,nullptr,"identity","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error);
  if(FAILED(hr)&&error)std::cerr<<static_cast<const char*>(error->GetBufferPointer());check(hr,"compile offline identity variant");
  check(dev.d->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&variants[v]),"variant shader");
 }
 std::ofstream csv(argv[2]);csv<<"width,height,ten,kind,round,groups,scene_gpu_ms\n";unsigned parity=0;
 for(auto cfg:std::vector<std::array<int,3>>{{64,36,0},{1920,1080,0},{1920,1080,1},{3840,2160,1},{7680,4320,1},{1080,1920,1}}){
  const int w=cfg[0],h=cfg[1];const bool ten=cfg[2];auto a=pattern(w,h,0,ten);auto fa=dev.upload(a,w,h,0,ten);
  for(int kind=0;kind<7;++kind){auto b=kind==1?pattern(w,h,8,ten):a;
   if(kind>=2){size_t i=kind==2?size_t(w)*h-1:kind==3?size_t(w)*h:kind==4?size_t(w)*h*3/2-1:kind==5?1:size_t(w)*(h/2)+w/2+1;if(ten)reinterpret_cast<uint16_t*>(b.data())[i]^=kind==6?1:64;else b[i]^=1;}
   auto fb=dev.upload(b,w,h,400000,ten);scene.bind(fa,fb,ten);scene.dispatch(w,h,false);auto reference=scene.read();
   for(int v=0;v<4;++v){scene.identity=variants[v];scene.identity_groups=groups[v];scene.dispatch(w,h,true);require(scene.read()==reference,"dispatch coverage/statistics mismatch");++parity;}
   if(w>=1920&&(kind==0||kind==1||kind==4))for(int round=0;round<8;++round)for(int j=0;j<4;++j){const int v=round%2?3-j:j;scene.identity=variants[v];scene.identity_groups=groups[v];csv<<w<<','<<h<<','<<ten<<','<<kind<<','<<round<<','<<groups[v]<<','<<scene.bench(w,h,true)<<'\n';}
   csv.flush();
  }
 }
 std::cout<<"PASS exactSceneStatistics="<<parity<<" groups=64/256/1024/4096 complete Y/UV/LSB coverage and grid early-out\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
