// Product validation. No filter registration or player configuration changes.
#include "nvof/gpu_engine.hpp"
#include <dxgi1_2.h>
#include <windows.h>
#include <mmsystem.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
using namespace nvof;
using Microsoft::WRL::ComPtr;
using Clock=std::chrono::steady_clock;
void require(bool v,const char* msg){if(!v)throw std::runtime_error(msg);}
void check(HRESULT h,const char* msg){if(FAILED(h))throw std::runtime_error(std::string(msg)+" HRESULT="+std::to_string(unsigned(h)));}
struct Device {
 ComPtr<ID3D11Device> d;ComPtr<ID3D11DeviceContext> c;
 Device(){ComPtr<IDXGIFactory1> f;check(CreateDXGIFactory1(IID_PPV_ARGS(&f)),"DXGI");ComPtr<IDXGIAdapter1> a;
  for(UINT i=0;;++i){ComPtr<IDXGIAdapter1> candidate;if(f->EnumAdapters1(i,&candidate)==DXGI_ERROR_NOT_FOUND)break;DXGI_ADAPTER_DESC1 desc{};candidate->GetDesc1(&desc);if(desc.VendorId==0x10de){a=candidate;break;}}
  require(bool(a),"NVIDIA D3D11/CUDA GPU required");const D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
  check(D3D11CreateDevice(a.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,D3D11_CREATE_DEVICE_VIDEO_SUPPORT,levels,2,D3D11_SDK_VERSION,&d,nullptr,&c),"D3D11 device");
 }
 GpuFrame upload(const std::vector<uint8_t>& bytes,int w,int h,int64_t pts,bool ten=false,unsigned transfer=0){
  D3D11_TEXTURE2D_DESC desc{};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;desc.Format=ten?DXGI_FORMAT_P010:DXGI_FORMAT_NV12;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA data{bytes.data(),UINT(w*(ten?2:1)),0};ComPtr<ID3D11Texture2D> t;check(d->CreateTexture2D(&desc,&data,&t),"Upload fixture");
  GpuFrame frame{t,0,w,h,pts};frame.hdr_transfer=transfer;return frame;
 }
 std::vector<uint8_t> read(const GpuFrame& frame){
  D3D11_TEXTURE2D_DESC desc{};frame.texture->GetDesc(&desc);desc.ArraySize=1;desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.MiscFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
  const unsigned stride=frame.width*(desc.Format==DXGI_FORMAT_P010?2:1);ComPtr<ID3D11Texture2D> t;check(d->CreateTexture2D(&desc,nullptr,&t),"Readback texture");
  c->CopySubresourceRegion(t.Get(),0,0,0,0,frame.texture.Get(),frame.array_slice,nullptr);D3D11_MAPPED_SUBRESOURCE m{};check(c->Map(t.Get(),0,D3D11_MAP_READ,0,&m),"Readback");
  std::vector<uint8_t> data(size_t(stride)*frame.height*3/2);for(int y=0;y<frame.height*3/2;++y)memcpy(data.data()+size_t(y)*stride,static_cast<uint8_t*>(m.pData)+size_t(y)*m.RowPitch,stride);c->Unmap(t.Get(),0);return data;
 }
 void wait(){D3D11_QUERY_DESC q{D3D11_QUERY_EVENT,0};ComPtr<ID3D11Query> event;check(d->CreateQuery(&q,&event),"Completion query");c->End(event.Get());c->Flush();auto start=Clock::now();
  while(c->GetData(event.Get(),nullptr,0,0)==S_FALSE){require(std::chrono::duration<double>(Clock::now()-start).count()<15,"GPU timeout");SwitchToThread();}}
};
std::vector<uint8_t> pattern(int w,int h,float shift,bool ten){
 std::vector<uint8_t> result(size_t(w)*h*3/2*(ten?2:1));
 auto put=[&](size_t i,int v){if(ten)reinterpret_cast<uint16_t*>(result.data())[i]=uint16_t(v*4+((i%7)==0?1:0))<<6;else result[i]=uint8_t(v);};
 for(int y=0;y<h;++y)for(int x=0;x<w;++x){float sx=x-shift;int v=int(126+40*std::sin(sx*.051)+35*std::cos(y*.081)+20*std::sin((sx+y)*.037));put(size_t(y)*w+x,std::clamp(v,16,235));}
 for(int y=0;y<h/2;++y)for(int x=0;x<w;x+=2){auto i=size_t(w)*h+size_t(y)*w+x;put(i,int(128+24*std::sin((x-shift)*.04)));put(i+1,int(128+18*std::cos(y*.06)));}
 return result;
}
std::unique_ptr<GpuRifeEngine> engine(Device& dev,const std::filesystem::path& runtime){return std::make_unique<GpuRifeEngine>(runtime,dev.d.Get(),dev.c.Get(),nullptr,GpuCompletionMode::context_ordered);}
double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void self_test(const std::filesystem::path& runtime,bool brief){
 Device dev;auto start=Clock::now();auto e=engine(dev,runtime);std::cout<<"constructor_ms="<<elapsed(start)<<'\n';require(MotionSynthesizer::engine_load_count()==0,"Constructor must not load neural plan before first original");
 std::vector<std::array<int,4>> cases=brief?std::vector<std::array<int,4>>{{640,360,0,0}}:std::vector<std::array<int,4>>{{1920,1080,0,0},{1280,720,0,0},{3840,2160,1,15},{1080,1920,0,0},{1024,1024,0,0},{2560,1080,0,0},{720,480,0,0},{640,360,1,16}};
 for(const auto& test:cases){int w=test[0],h=test[1];bool ten=test[2];unsigned transfer=test[3];
  auto a=pattern(w,h,0,ten),b=pattern(w,h,8,ten);auto fa=dev.upload(a,w,h,0,ten,transfer),fb=dev.upload(b,w,h,400000,ten,transfer);
  auto meta=std::make_shared<HdrMetadata>();meta->data[0].assign(80,37);fa.hdr=meta;
  start=Clock::now();auto original=e->copy(fa,true);double original_ms=elapsed(start);require(dev.read(original)==a,"Original source frame changed");
  auto first=e->interpolate_pair(fa,fb,{100000,200000,300000});require(first.frames.size()==3&&first.quality.native_synthesized_mask==7,"RIFE did not synthesize every requested phase");
  std::vector<std::vector<uint8_t>> reference;for(auto& f:first.frames){require(f.hdr==meta&&f.hdr_transfer==transfer,"HDR metadata lost");reference.push_back(dev.read(f));
   if(ten)for(size_t k=0;k<reference.back().size();k+=2)require((reference.back()[k]&63)==0,"P010 output has nonzero low bits");}
  require(reference[1]!=a&&reference[1]!=b,"Midpoint was held original");
  double reset_max=0;for(int i=0;i<(brief?1:6);++i){start=Clock::now();e->reset();reset_max=(std::max)(reset_max,elapsed(start));auto repeated=e->interpolate_pair(fa,fb,{100000,200000,300000});
   for(int k=0;k<3;++k)require(dev.read(repeated.frames[k])==reference[k],"Seek changed output or leaked previous image history");}
  auto same=fa;same.pts=400000;uint64_t calls=MotionSynthesizer::inference_count();auto still=e->interpolate_pair(fa,same,{200000});require(still.frames.size()==1&&dev.read(still.frames[0])==a&&MotionSynthesizer::inference_count()==calls,"Static bypass failed");
  std::cout<<"PASS "<<w<<'x'<<h<<" ten_bit="<<ten<<" transfer="<<transfer<<" original_ms="<<original_ms<<" reset_max_ms="<<reset_max<<" loads="<<MotionSynthesizer::engine_load_count()<<std::endl;
  e.reset();e=engine(dev,runtime); // File open: new context, shared immutable plan.
 }
 require(MotionSynthesizer::engine_load_count()==1,"Plan deserialized again across seeks/files");require(!GetModuleHandleW(L"nvofapi64.dll"),"RIFE main unexpectedly loaded NvOF");
 require(GetModuleHandleW(L"nvinfer_10.dll")&&GetModuleHandleW(L"cudart64_13.dll"),"TensorRT/CUDA was not active");
 std::cout<<"PASS engine_loads=1 no_nvof=1 runtime_compiler=absent inference_calls="<<MotionSynthesizer::inference_count()<<std::endl;
}
std::vector<uint8_t> load(const std::filesystem::path& path,size_t size){std::ifstream f(path,std::ios::binary);require(bool(f),"Input frame missing");std::vector<uint8_t> b(size);f.read(reinterpret_cast<char*>(b.data()),b.size());require(size_t(f.gcount())==size,"Input frame size mismatch");return b;}
void pair(const std::filesystem::path& runtime,int w,int h,const std::filesystem::path& pa,const std::filesystem::path& pb,const std::filesystem::path& prefix){
 Device dev;auto e=engine(dev,runtime);auto a=dev.upload(load(pa,size_t(w)*h*3/2),w,h,0),b=dev.upload(load(pb,size_t(w)*h*3/2),w,h,400000);
 auto output=e->interpolate_pair(a,b,{100000,200000,300000});require(output.frames.size()==3,"Pair rejected by scene gate");
 for(int k=0;k<3;++k){auto raw=dev.read(output.frames[k]);std::ofstream f(prefix.wstring()+L"-"+std::to_wstring(k+1)+L".nv12",std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),raw.size());require(bool(f),"Output frame write failed");}
 for(int phases:{1,3}){std::vector<int64_t> pts=phases==1?std::vector<int64_t>{200000}:std::vector<int64_t>{100000,200000,300000};std::vector<double> ms;
  for(int i=0;i<70;++i){auto start=Clock::now();auto r=e->interpolate_pair(a,b,pts);dev.wait();if(i>=10)ms.push_back(elapsed(start));}
  auto sum=std::accumulate(ms.begin(),ms.end(),0.0);std::sort(ms.begin(),ms.end());std::cout<<"BENCH phases="<<phases<<" mean_ms="<<sum/ms.size()<<" median_ms="<<ms[ms.size()/2]<<" p95_ms="<<ms[size_t(ms.size()*.95)]<<'\n';}
 std::cout<<"PASS loads="<<MotionSynthesizer::engine_load_count()<<" inferences="<<MotionSynthesizer::inference_count()<<std::endl;
}
void check_optimized(const std::filesystem::path& runtime){
 Device dev;
 SetEnvironmentVariableW(L"RIFE_HEAD_CACHE_SYNC_PREPARE",L"1");
 for(auto spec:std::vector<std::array<int,3>>{{640,360,0},{3840,1604,1}}){
  const int w=spec[0],h=spec[1];const bool ten=spec[2];
  auto a=dev.upload(pattern(w,h,0,ten),w,h,0,ten,ten?15:0);
  auto b=dev.upload(pattern(w,h,8,ten),w,h,400000,ten,ten?15:0);
  SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",L"0");
  SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PACKED_MOTION",L"0");
  auto baseline=engine(dev,runtime);auto expected=baseline->interpolate_pair(a,b,{100000,200000,300000});
  SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",L"1");
  SetEnvironmentVariableW(L"RIFE_EXPERIMENT_PACKED_MOTION",L"1");
  auto optimized=engine(dev,runtime);optimized->enable_timing(true,true,1);
  const auto heads=MotionSynthesizer::head_cache_pair_count();
  auto actual=optimized->interpolate_pair(a,b,{100000,200000,300000});dev.wait();
  require(MotionSynthesizer::head_cache_pair_count()==heads+1,"E03 engines unavailable or incompatible");
  require(actual.frames.size()==3&&expected.frames.size()==3,"Missing optimized outputs");
  auto timing=optimized->pair_timing();require(timing.rife&&timing.rife->phase_count==3,"Optimization timing unavailable");
  for(size_t i=0;i<3;++i){
   require(timing.rife->phases[i].head_cached&&timing.rife->phases[i].packed_layout==(w>=3840),"Unexpected E03/E04 policy");
   require(dev.read(actual.frames[i])==dev.read(expected.frames[i]),"Optimized engine output differs from the original engine");
  }
  std::cout<<"PASS optimized engines "<<w<<'x'<<h<<" exactOutputs=3 E03=1 E04="<<(w>=3840)<<std::endl;
 }
}
int wmain(int argc,wchar_t** argv){timeBeginPeriod(1);int code=0;try{
 require(argc>=3,"Usage: RifeEngineCheck --check|--self-test runtime; --pair runtime width height a.nv12 b.nv12 output-prefix");
 if(std::wstring(argv[1])==L"--check-optimized"){require(argc==3,"Invalid optimized check arguments");check_optimized(argv[2]);}
 else if(std::wstring(argv[1])==L"--pair"){require(argc==8,"Invalid pair arguments");pair(argv[2],std::stoi(argv[3]),std::stoi(argv[4]),argv[5],argv[6],argv[7]);}
 else{require(std::wstring(argv[1])==L"--check"||std::wstring(argv[1])==L"--self-test","Invalid probe mode");self_test(argv[2],std::wstring(argv[1])==L"--check");}
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;code=1;}timeEndPeriod(1);return code;}
