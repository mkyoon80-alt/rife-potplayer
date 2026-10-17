// Offline E04 screening only: full native Y/UV composition including every
// packing dispatch. No change to the player shader or inference path.
#define wmain unused_layout_rife_main
#include "rife_probe.cpp"
#undef wmain
#include <d3dcompiler.h>
#include <limits>
#include <iomanip>
struct LayoutProbe {
 static constexpr unsigned W=960,H=540,N=W*H;
 Device& dev;ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps[2][2];ComPtr<ID3D11ComputeShader> pack;
 ComPtr<ID3D11Buffer> motion,flow,mask,cb;ComPtr<ID3D11ShaderResourceView> raw,floatRaw,flowView,maskView;
 ComPtr<ID3D11UnorderedAccessView> flowOut,maskOut;ComPtr<ID3D11SamplerState> sampler;ComPtr<ID3D11RasterizerState> raster;
 std::array<ComPtr<ID3D11ShaderResourceView>,4> views;ComPtr<ID3D11RenderTargetView> targets[2];GpuFrame output;
 struct Params{float w,h;unsigned gw=W,gh=H;float phase=.5f;unsigned plane=0,ten=0,hdr=0;float aw=W,ah=H,ox=0,oy=0;unsigned portrait=0;float unused[3]{};} p{};
 ComPtr<ID3DBlob> compile(const std::string& s,const char* entry,const char* target){ComPtr<ID3DBlob> b,error;auto hr=D3DCompile(s.data(),s.size(),"offline-layout",nullptr,nullptr,entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&b,&error);if(FAILED(hr)&&error)std::cerr<<static_cast<const char*>(error->GetBufferPointer());check(hr,"layout compile");return b;}
 void buffer(ComPtr<ID3D11Buffer>& b,unsigned bytes,unsigned bind){D3D11_BUFFER_DESC d{};d.ByteWidth=bytes;d.BindFlags=bind;check(dev.d->CreateBuffer(&d,nullptr,&b),"layout buffer");}
 void srv(ID3D11Buffer* b,DXGI_FORMAT format,unsigned count,ComPtr<ID3D11ShaderResourceView>& view){D3D11_SHADER_RESOURCE_VIEW_DESC d{};d.Format=format;d.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;d.Buffer.NumElements=count;check(dev.d->CreateShaderResourceView(b,&d,&view),"layout srv");}
 void uav(ID3D11Buffer* b,DXGI_FORMAT format,unsigned count,ComPtr<ID3D11UnorderedAccessView>& view){D3D11_UNORDERED_ACCESS_VIEW_DESC d{};d.Format=format;d.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;d.Buffer.NumElements=count;check(dev.d->CreateUnorderedAccessView(b,&d,&view),"layout uav");}
 LayoutProbe(Device& d,std::string source,const std::filesystem::path& out):dev(d){
  auto code=compile(source,"vs","vs_5_0");check(dev.d->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vs),"layout vertex");
  for(int v=0;v<2;++v){auto s=source;if(v){
   auto replace=[&](const std::string& from,const std::string& to){auto pos=s.find(from);require(pos!=std::string::npos,"layout shader contract");s.replace(pos,from.size(),to);};
   replace("Buffer<float> motion:register(t4);","Buffer<float4> packedMotion:register(t4); Buffer<float> packedMask:register(t5);");
   replace("float4 v=float4(motion[i],motion[n+i],motion[2*n+i],motion[3*n+i]);","float4 v=packedMotion[i];");replace("float m=saturate(motion[4*n+i]);","float m=saturate(packedMask[i]);");
  }
  for(int plane=0;plane<2;++plane){auto b=compile(s,plane?"compositeUV":"compositeY","ps_5_0");check(dev.d->CreatePixelShader(b->GetBufferPointer(),b->GetBufferSize(),nullptr,&ps[v][plane]),"layout pixel");
   if(!plane){ComPtr<ID3DBlob> text;check(D3DDisassemble(b->GetBufferPointer(),b->GetBufferSize(),0,nullptr,&text),"disassemble");std::ofstream file(out/(v?L"packed-isa.txt":L"original-isa.txt"));file.write(static_cast<const char*>(text->GetBufferPointer()),text->GetBufferSize());}
  }}
  auto b=compile("Buffer<uint> source:register(t4); RWBuffer<uint4> flow:register(u0); RWBuffer<uint> mask:register(u1); [numthreads(256,1,1)] void main(uint i:SV_DispatchThreadID){if(i<518400){flow[i]=uint4(source[i],source[518400+i],source[1036800+i],source[1555200+i]);mask[i]=source[2073600+i];}}","main","cs_5_0");check(dev.d->CreateComputeShader(b->GetBufferPointer(),b->GetBufferSize(),nullptr,&pack),"layout pack");
  buffer(motion,N*20,D3D11_BIND_SHADER_RESOURCE);buffer(flow,N*16,D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS);buffer(mask,N*4,D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS);buffer(cb,sizeof(p),D3D11_BIND_CONSTANT_BUFFER);
  srv(motion.Get(),DXGI_FORMAT_R32_UINT,N*5,raw);srv(motion.Get(),DXGI_FORMAT_R32_FLOAT,N*5,floatRaw);srv(flow.Get(),DXGI_FORMAT_R32G32B32A32_FLOAT,N,flowView);srv(mask.Get(),DXGI_FORMAT_R32_FLOAT,N,maskView);uav(flow.Get(),DXGI_FORMAT_R32G32B32A32_UINT,N,flowOut);uav(mask.Get(),DXGI_FORMAT_R32_UINT,N,maskOut);
  D3D11_SAMPLER_DESC sd{};sd.Filter=D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;sd.AddressU=sd.AddressV=sd.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;sd.MaxLOD=D3D11_FLOAT32_MAX;check(dev.d->CreateSamplerState(&sd,&sampler),"sampler");D3D11_RASTERIZER_DESC rd{};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;rd.DepthClipEnable=TRUE;check(dev.d->CreateRasterizerState(&rd,&raster),"raster");
 }
 void bind(const GpuFrame&a,const GpuFrame&b,bool ten,unsigned hdr){
  p.w=float(a.width);p.h=float(a.height);p.ten=ten;p.hdr=hdr;p.portrait=a.height>a.width;p.aw=W;p.ah=H;p.ox=p.oy=0;
  dev.c->UpdateSubresource(cb.Get(),0,nullptr,&p,0,0);
  for(int i=0;i<4;++i){views[i].Reset();D3D11_SHADER_RESOURCE_VIEW_DESC d{};d.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D;d.Texture2D.MipLevels=1;d.Format=i<2?(ten?DXGI_FORMAT_R16_UNORM:DXGI_FORMAT_R8_UNORM):(ten?DXGI_FORMAT_R16G16_UNORM:DXGI_FORMAT_R8G8_UNORM);check(dev.d->CreateShaderResourceView(i%2?b.texture.Get():a.texture.Get(),&d,&views[i]),"layout source plane");}
  output={};for(auto&t:targets)t.Reset();D3D11_TEXTURE2D_DESC td{};td.Width=a.width;td.Height=a.height;td.ArraySize=td.MipLevels=td.SampleDesc.Count=1;td.Format=ten?DXGI_FORMAT_P010:DXGI_FORMAT_NV12;td.BindFlags=D3D11_BIND_RENDER_TARGET;check(dev.d->CreateTexture2D(&td,nullptr,&output.texture),"layout target");output.width=a.width;output.height=a.height;
  for(int i=0;i<2;++i){D3D11_RENDER_TARGET_VIEW_DESC d{};d.ViewDimension=D3D11_RTV_DIMENSION_TEXTURE2D;d.Format=i?(ten?DXGI_FORMAT_R16G16_UNORM:DXGI_FORMAT_R8G8_UNORM):(ten?DXGI_FORMAT_R16_UNORM:DXGI_FORMAT_R8_UNORM);check(dev.d->CreateRenderTargetView(output.texture.Get(),&d,&targets[i]),"layout output plane");}
 }
 void draw(bool packed){
  if(packed){auto r=raw.Get();dev.c->CSSetShaderResources(4,1,&r);ID3D11UnorderedAccessView* u[]={flowOut.Get(),maskOut.Get()};dev.c->CSSetUnorderedAccessViews(0,2,u,nullptr);dev.c->CSSetShader(pack.Get(),nullptr,0);dev.c->Dispatch((N+255)/256,1,1);r=nullptr;dev.c->CSSetShaderResources(4,1,&r);u[0]=u[1]=nullptr;dev.c->CSSetUnorderedAccessViews(0,2,u,nullptr);dev.c->CSSetShader(nullptr,nullptr,0);}
  dev.c->IASetInputLayout(nullptr);dev.c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);dev.c->VSSetShader(vs.Get(),nullptr,0);dev.c->RSSetState(raster.Get());dev.c->OMSetBlendState(nullptr,nullptr,~0u);dev.c->OMSetDepthStencilState(nullptr,0);
  ID3D11ShaderResourceView* v[]={views[0].Get(),views[1].Get(),views[2].Get(),views[3].Get(),packed?flowView.Get():floatRaw.Get(),packed?maskView.Get():nullptr};dev.c->PSSetShaderResources(0,6,v);auto s=sampler.Get();dev.c->PSSetSamplers(0,1,&s);auto c=cb.Get();dev.c->PSSetConstantBuffers(0,1,&c);
  for(int i=0;i<2;++i){dev.c->PSSetShader(ps[packed][i].Get(),nullptr,0);auto t=targets[i].Get();dev.c->OMSetRenderTargets(1,&t,nullptr);D3D11_VIEWPORT vp{0,0,p.w/(i?2:1),p.h/(i?2:1),0,1};dev.c->RSSetViewports(1,&vp);dev.c->Draw(3,0);}ID3D11ShaderResourceView* empty[6]{};dev.c->PSSetShaderResources(0,6,empty);dev.c->OMSetRenderTargets(0,nullptr,nullptr);
 }
 double bench(bool packed){D3D11_QUERY_DESC q{D3D11_QUERY_TIMESTAMP_DISJOINT,0};ComPtr<ID3D11Query> f,a,b;check(dev.d->CreateQuery(&q,&f),"freq");q.Query=D3D11_QUERY_TIMESTAMP;check(dev.d->CreateQuery(&q,&a),"time");check(dev.d->CreateQuery(&q,&b),"time");for(int i=0;i<5;++i)draw(packed);dev.wait();dev.c->Begin(f.Get());dev.c->End(a.Get());for(int i=0;i<32;++i)draw(packed);dev.c->End(b.Get());dev.c->End(f.Get());dev.wait();D3D11_QUERY_DATA_TIMESTAMP_DISJOINT freq{};UINT64 aa=0,bb=0;check(dev.c->GetData(f.Get(),&freq,sizeof(freq),0),"freq read");check(dev.c->GetData(a.Get(),&aa,sizeof(aa),0),"time read");check(dev.c->GetData(b.Get(),&bb,sizeof(bb),0),"time read");require(!freq.Disjoint&&freq.Frequency,"invalid timer");return double(bb-aa)*1000/freq.Frequency/32;}
};
int wmain(int argc,wchar_t**argv){try{require(argc==3,"shader output-folder");std::ifstream f(argv[1]);std::string source((std::istreambuf_iterator<char>(f)),{});Device dev;LayoutProbe p(dev,source,argv[2]);std::ofstream csv(std::filesystem::path(argv[2])/L"layout-cost.csv");csv<<"width,height,ten,hdr,round,packed,gpu_ms\n"<<std::setprecision(12);unsigned exact=0;
 for(auto spec:std::vector<std::array<int,4>>{{1920,1080,0,0},{1920,1080,1,0},{3840,2160,1,15},{7680,4320,1,16},{1080,1920,1,0},{2560,1080,0,0}}){int w=spec[0],h=spec[1];auto a=dev.upload(pattern(w,h,0,spec[2]),w,h,0,spec[2]),b=dev.upload(pattern(w,h,8,spec[2]),w,h,400000,spec[2]);p.bind(a,b,spec[2],spec[3]);
  for(int variant=0;variant<5;++variant){std::vector<float> values(LayoutProbe::N*5);for(unsigned i=0;i<LayoutProbe::N;++i){for(unsigned c=0;c<4;++c)values[LayoutProbe::N*c+i]=float(variant+1)*12.f*std::sin(float(i+c*123)/137.f);values[LayoutProbe::N*4+i]=float((i*7)%997)/996.f;}if(variant==4){values[0]=std::numeric_limits<float>::quiet_NaN();values[1]=std::numeric_limits<float>::infinity();values[LayoutProbe::N*4+5]=std::numeric_limits<float>::quiet_NaN();}
   dev.c->UpdateSubresource(p.motion.Get(),0,nullptr,values.data(),0,0);p.draw(false);auto ref=dev.read(p.output);p.draw(true);require(ref==dev.read(p.output),"layout changed final Y/UV pixels");++exact;
   if(variant==2)for(int round=0;round<8;++round)for(int order=0;order<2;++order){bool packed=(round+order)%2;csv<<w<<','<<h<<','<<spec[2]<<','<<spec[3]<<','<<round<<','<<packed<<','<<p.bench(packed)<<'\n';}
  }csv.flush();
 }
 std::cout<<"PASS synthetic map screening exactOutputs="<<exact<<" packing included, FP32 vector flow+mask, four candidates, NaN/Inf, NV12/P010/SDR/PQ/HLG/8K/portrait/ultrawide. Not a production inference or real-video benchmark.\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
