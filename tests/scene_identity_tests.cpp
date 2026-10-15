
#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "scene_shaders.hpp"
#include <array>
struct CheckScene {
 Device& dev;ComPtr<ID3D11ComputeShader> legacy,grid,identity,match;ComPtr<ID3D11Buffer> stats,readback,constants;ComPtr<ID3D11UnorderedAccessView> uav;
 CheckScene(Device& d):dev(d){
  check(dev.d->CreateComputeShader(shaders::scene::main,sizeof(shaders::scene::main),nullptr,&legacy),"legacy");
  check(dev.d->CreateComputeShader(shaders::scene::gridProbe,sizeof(shaders::scene::gridProbe),nullptr,&grid),"grid");
  check(dev.d->CreateComputeShader(shaders::scene::identity,sizeof(shaders::scene::identity),nullptr,&identity),"identity");
  check(dev.d->CreateComputeShader(shaders::scene::match,sizeof(shaders::scene::match),nullptr,&match),"match");
  D3D11_BUFFER_DESC bd{};bd.ByteWidth=(81+64*36)*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;check(dev.d->CreateBuffer(&bd,nullptr,&stats),"stats");D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=bd.ByteWidth/4;check(dev.d->CreateUnorderedAccessView(stats.Get(),&ud,&uav),"uav");
  bd.ByteWidth=81*4;bd.Usage=D3D11_USAGE_STAGING;bd.BindFlags=0;bd.MiscFlags=0;bd.StructureByteStride=0;bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;check(dev.d->CreateBuffer(&bd,nullptr,&readback),"readback");bd={};bd.ByteWidth=16;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;check(dev.d->CreateBuffer(&bd,nullptr,&constants),"constants");
 }
 std::array<ComPtr<ID3D11ShaderResourceView>,4> views;
 void bind(const GpuFrame& a,const GpuFrame& b,bool ten){
  uint32_t dims[4]={uint32_t(a.width),uint32_t(a.height),0,0};dev.c->UpdateSubresource(constants.Get(),0,nullptr,dims,0,0);
  for(int i=0;i<4;++i){views[i].Reset();D3D11_SHADER_RESOURCE_VIEW_DESC sd{};sd.Format=i<2?(ten?DXGI_FORMAT_R16_UNORM:DXGI_FORMAT_R8_UNORM):(ten?DXGI_FORMAT_R16G16_UNORM:DXGI_FORMAT_R8G8_UNORM);sd.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D;sd.Texture2D.MipLevels=1;check(dev.d->CreateShaderResourceView(i%2?b.texture.Get():a.texture.Get(),&sd,&views[i]),"plane");}
 }
 void dispatch(int w,int h,bool fast){
  const UINT zero[4]={};dev.c->ClearUnorderedAccessViewUint(uav.Get(),zero);ID3D11ShaderResourceView* v[]={views[0].Get(),views[1].Get(),views[2].Get(),views[3].Get()};ID3D11UnorderedAccessView* u=uav.Get();ID3D11Buffer* cb=constants.Get();dev.c->CSSetShaderResources(0,4,v);dev.c->CSSetUnorderedAccessViews(0,1,&u,nullptr);dev.c->CSSetConstantBuffers(0,1,&cb);
  dev.c->CSSetShader(fast?grid.Get():legacy.Get(),nullptr,0);dev.c->Dispatch(fast?4:(std::max(w,64)+15)/16,fast?3:(std::max(h,36)+15)/16,1);
  if(fast){dev.c->CSSetShader(identity.Get(),nullptr,0);dev.c->Dispatch(4096,1,1);}
  dev.c->CSSetShader(match.Get(),nullptr,0);dev.c->Dispatch(80,1,1);u=nullptr;dev.c->CSSetUnorderedAccessViews(0,1,&u,nullptr);ID3D11ShaderResourceView* empty[4]={};dev.c->CSSetShaderResources(0,4,empty);
 }
 std::array<uint32_t,81> read(){D3D11_BOX box{0,0,0,81*4,1,1};dev.c->CopySubresourceRegion(readback.Get(),0,0,0,0,stats.Get(),0,&box);D3D11_MAPPED_SUBRESOURCE m{};check(dev.c->Map(readback.Get(),0,D3D11_MAP_READ,0,&m),"map");std::array<uint32_t,81> r;memcpy(r.data(),m.pData,sizeof(r));dev.c->Unmap(readback.Get(),0);return r;}
 double bench(int w,int h,bool fast){
  D3D11_QUERY_DESC q{D3D11_QUERY_TIMESTAMP_DISJOINT,0};ComPtr<ID3D11Query> freq,a,b;check(dev.d->CreateQuery(&q,&freq),"frequency");q.Query=D3D11_QUERY_TIMESTAMP;check(dev.d->CreateQuery(&q,&a),"timestamp a");check(dev.d->CreateQuery(&q,&b),"timestamp b");
  for(int i=0;i<5;++i)dispatch(w,h,fast);dev.wait();dev.c->Begin(freq.Get());dev.c->End(a.Get());for(int i=0;i<40;++i)dispatch(w,h,fast);dev.c->End(b.Get());dev.c->End(freq.Get());dev.wait();D3D11_QUERY_DATA_TIMESTAMP_DISJOINT f{};UINT64 aa=0,bb=0;require(dev.c->GetData(freq.Get(),&f,sizeof(f),0)==S_OK&&!f.Disjoint&&f.Frequency,"frequency invalid");check(dev.c->GetData(a.Get(),&aa,sizeof(aa),0),"timestamp a result");check(dev.c->GetData(b.Get(),&bb,sizeof(bb),0),"timestamp b result");return (bb-aa)*1000.0/f.Frequency/40;
 }
};
int main(){try{Device dev;CheckScene scene(dev);int cases=0;
 for(const auto cfg:std::vector<std::array<int,3>>{{64,36,0},{1920,1080,0},{1080,1920,1},{7680,4320,1}}){int w=cfg[0],h=cfg[1];bool ten=cfg[2];auto a=pattern(w,h,0,ten);auto fa=dev.upload(a,w,h,0,ten);
  for(int kind=0;kind<7;++kind){auto b=kind==1?pattern(w,h,8,ten):a;
   if(kind>=2){size_t i=kind==2?size_t(w)*h-1:kind==3?size_t(w)*h:kind==4?size_t(w)*h*3/2-1:kind==5?1:size_t(w)*(h/2)+w/2+1;if(ten)reinterpret_cast<uint16_t*>(b.data())[i]^=kind==6?1:64;else b[i]^=1;}
   auto fb=dev.upload(b,w,h,400000,ten);scene.bind(fa,fb,ten);scene.dispatch(w,h,false);auto ref=scene.read();scene.dispatch(w,h,true);auto test=scene.read();require(ref==test,"scene aggregate or exact identity differs");require(test[80]==uint32_t(kind!=0),"off-grid or chroma difference was missed");++cases;
   if(w==7680&&(kind==0||kind==1)){auto old1=scene.bench(w,h,false),new1=scene.bench(w,h,true),new2=scene.bench(w,h,true),old2=scene.bench(w,h,false);std::cout<<"BENCH 8k_p010 kind="<<kind<<" before_gpu_ms="<<(old1+old2)/2<<" after_gpu_ms="<<(new1+new2)/2<<std::endl;}
  }
 }
 std::cout<<"PASS exact statistics and identity parity cases="<<cases<<" (still, moving, off-grid Y, first/last UV, P010 LSB, portrait, 8K)\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
