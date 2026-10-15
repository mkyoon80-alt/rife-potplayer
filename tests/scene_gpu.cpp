#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include "scene_fixture.hpp"
#include "scene_shaders.hpp"
int main(int argc,char**argv){try{
 require(argc==3,"scene_gpu fixture.bin results.csv");Device dev;auto fixtures=read_scene_fixtures(argv[1]);
 ComPtr<ID3D11ComputeShader> first,identity,match;check(dev.d->CreateComputeShader(shaders::scene::identity,sizeof(shaders::scene::identity),nullptr,&identity),"identity shader");check(dev.d->CreateComputeShader(shaders::scene::gridProbe,sizeof(shaders::scene::gridProbe),nullptr,&first),"scene shader");check(dev.d->CreateComputeShader(shaders::scene::match,sizeof(shaders::scene::match),nullptr,&match),"scene match");
 D3D11_BUFFER_DESC bd{};bd.ByteWidth=(scene_grid_offset+scene_grid_count)*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;
 ComPtr<ID3D11Buffer> stats,readback,constants;check(dev.d->CreateBuffer(&bd,nullptr,&stats),"scene stats");D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=bd.ByteWidth/4;ComPtr<ID3D11UnorderedAccessView> uav;check(dev.d->CreateUnorderedAccessView(stats.Get(),&ud,&uav),"scene uav");
 bd.ByteWidth=scene_grid_offset*4;bd.Usage=D3D11_USAGE_STAGING;bd.BindFlags=0;bd.MiscFlags=0;bd.StructureByteStride=0;bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;check(dev.d->CreateBuffer(&bd,nullptr,&readback),"scene readback");
 bd={};bd.ByteWidth=16;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;check(dev.d->CreateBuffer(&bd,nullptr,&constants),"scene constants");uint32_t dims[4]={128,72,0,0};dev.c->UpdateSubresource(constants.Get(),0,nullptr,dims,0,0);
 std::vector<uint8_t> raw(128*72*3/2,128);auto ta=dev.upload(raw,128,72,0),tb=dev.upload(raw,128,72,400000);ComPtr<ID3D11ShaderResourceView> views[4];
 for(int i=0;i<4;++i){D3D11_SHADER_RESOURCE_VIEW_DESC sd{};sd.Format=i<2?DXGI_FORMAT_R8_UNORM:DXGI_FORMAT_R8G8_UNORM;sd.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D;sd.Texture2D.MipLevels=1;check(dev.d->CreateShaderResourceView(i%2?tb.texture.Get():ta.texture.Get(),&sd,&views[i]),"scene plane");}
 auto upload=[&](const SceneGrid& grid,ID3D11Texture2D* texture){for(unsigned y=0;y<72;++y)for(unsigned x=0;x<128;++x)raw[y*128+x]=grid[(y/2)*64+x/2];dev.c->UpdateSubresource(texture,0,nullptr,raw.data(),128,0);};
 std::ofstream csv(argv[2]);csv<<"window,frame,label,cut,normalized,edge,patch_mean,patch_fraction,baseline\n";unsigned total=0,labels=0,cuts=0,wrong=0,parity=0;
 for(auto& f:fixtures){SceneDetector detector,cpu;for(size_t i=1;i<f.frames.size();++i){upload(f.frames[i-1],ta.texture.Get());upload(f.frames[i],tb.texture.Get());const UINT zero[4]={};dev.c->ClearUnorderedAccessViewUint(uav.Get(),zero);ID3D11ShaderResourceView* srvs[]={views[0].Get(),views[1].Get(),views[2].Get(),views[3].Get()};ID3D11UnorderedAccessView* uv=uav.Get();ID3D11Buffer* cb=constants.Get();dev.c->CSSetShaderResources(0,4,srvs);dev.c->CSSetUnorderedAccessViews(0,1,&uv,nullptr);dev.c->CSSetConstantBuffers(0,1,&cb);dev.c->CSSetShader(first.Get(),nullptr,0);dev.c->Dispatch(4,3,1);dev.c->CSSetShader(identity.Get(),nullptr,0);dev.c->Dispatch(4096,1,1);dev.c->CSSetShader(match.Get(),nullptr,0);dev.c->Dispatch(80,1,1);
   uv=nullptr;dev.c->CSSetUnorderedAccessViews(0,1,&uv,nullptr);ID3D11ShaderResourceView* empty[4]={};dev.c->CSSetShaderResources(0,4,empty);
   D3D11_BOX box{0,0,0,scene_grid_offset*4,1,1};dev.c->CopySubresourceRegion(readback.Get(),0,0,0,0,stats.Get(),0,&box);D3D11_MAPPED_SUBRESOURCE mapped{};check(dev.c->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped),"scene map");SceneStats s{};memcpy(&s,mapped.pData,sizeof(s));dev.c->Unmap(readback.Get(),0);auto d=detector.evaluate(s);
   auto ref=cpu.evaluate(scene_stats_grid(f.frames[i-1],f.frames[i]));if(d.cut!=ref.cut){++parity;std::cerr<<"PARITY "<<f.name<<' '<<i<<" gpu="<<d.cut<<" cpu="<<ref.cut<<" patch="<<d.patch_mean<<'/'<<ref.patch_mean<<'\n';}
   if(f.labels[i]>=0){++labels;if(d.cut!=bool(f.labels[i])){++wrong;std::cerr<<"WRONG "<<f.name<<' '<<i<<" label="<<f.labels[i]<<" gpu="<<d.cut<<'\n';}}
   ++total;cuts+=d.cut;csv<<f.name<<','<<i<<','<<f.labels[i]<<','<<d.cut<<','<<d.normalized<<','<<d.edge_correlation<<','<<d.patch_mean<<','<<d.patch_fraction<<','<<d.baseline<<'\n';
  }std::cout<<"WINDOW "<<f.name<<" pairs="<<f.frames.size()-1<<std::endl;}
 std::cout<<"RESULT pairs="<<total<<" labels="<<labels<<" detected="<<cuts<<" wrong="<<wrong<<" parity="<<parity<<std::endl;require(wrong==0&&parity==0,"Scene GPU corpus validation failed");return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
