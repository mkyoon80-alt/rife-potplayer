#define wmain unused_cost_probe_main
#include "rife_probe.cpp"
#undef wmain
#include <thread>
#include <iomanip>
#include <dxgi1_4.h>
int wmain(int argc,wchar_t** argv){timeBeginPeriod(1);try {
    require(argc==8,"runtime width height phases input_fps iterations output.csv");
    const int w=std::stoi(argv[2]),h=std::stoi(argv[3]),phases=std::stoi(argv[4]),fps=std::stoi(argv[5]),iterations=std::stoi(argv[6]);
    Device dev;auto e=engine(dev,argv[1]);
    auto fa=dev.upload(pattern(w,h,0,true),w,h,0,true),fb=dev.upload(pattern(w,h,8,true),w,h,10000000/fps,true);
    std::vector<int64_t> times;for(int i=1;i<=phases;++i)times.push_back(fb.pts*i/(phases+1));
    const auto coldStart=Clock::now();e->interpolate_pair(fa,fb,times);dev.wait();const auto coldMs=elapsed(coldStart);
    wchar_t cacheOption[8]{};const bool cacheRequested=GetEnvironmentVariableW(L"RIFE_EXPERIMENT_PAIR_HEAD_CACHE",cacheOption,8)>0&&cacheOption[0]==L'1';
    if(cacheRequested&&phases>1){
        auto prepareStart=Clock::now();while(MotionSynthesizer::head_cache_pair_count()==0){
            require(elapsed(prepareStart)<15000,"Head preparation unavailable");e->interpolate_pair(fa,fb,times);dev.wait();Sleep(1);
        }
    }
    for(int i=0;i<24;++i){e->interpolate_pair(fa,fb,times);dev.wait();}
    std::cerr<<"STARTUP firstPairMs="<<coldMs<<" warmedMs="<<elapsed(coldStart)<<'\n';
    D3D11_QUERY_DESC desc{D3D11_QUERY_TIMESTAMP_DISJOINT,0};ComPtr<ID3D11Query> disjoint,start,stop;
    check(dev.d->CreateQuery(&desc,&disjoint),"disjoint");desc.Query=D3D11_QUERY_TIMESTAMP;
    check(dev.d->CreateQuery(&desc,&start),"timestamp start");check(dev.d->CreateQuery(&desc,&stop),"timestamp stop");
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgi;Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
    DXGI_QUERY_VIDEO_MEMORY_INFO memory{};dev.d.As(&dxgi);dxgi->GetAdapter(&adapter);adapter.As(&adapter3);
    if(adapter3)adapter3->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&memory);
    struct Record {double wall,gpu;};std::vector<Record> records;records.reserve(iterations);
    // Driver energy counters are read by the parent before/after this handshake.
    // No decoder, display, VSR or physical presentation is included in this probe.
    std::cout<<"READY gpuBytes="<<memory.CurrentUsage<<std::endl;std::cin.get();
    const auto epoch=Clock::now();const auto inferenceBefore=MotionSynthesizer::inference_count();
    const auto headBefore=MotionSynthesizer::head_cache_pair_count(),phaseBefore=MotionSynthesizer::head_cache_phase_count();
    uint64_t outputs=0;unsigned deadlineMisses=0;
    for(int i=0;i<iterations;++i) {
        const auto begin=Clock::now();dev.c->Begin(disjoint.Get());dev.c->End(start.Get());
        auto batch=e->interpolate_pair(fa,fb,times);dev.c->End(stop.Get());dev.c->End(disjoint.Get());dev.c->Flush();
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT frequency{};
        HRESULT ready;while((ready=dev.c->GetData(disjoint.Get(),&frequency,sizeof(frequency),D3D11_ASYNC_GETDATA_DONOTFLUSH))==S_FALSE){require(elapsed(begin)<15000,"query timeout");SwitchToThread();}
        check(ready,"disjoint result");require(!frequency.Disjoint&&frequency.Frequency,"invalid GPU timestamp");
        UINT64 a=0,b=0;check(dev.c->GetData(start.Get(),&a,sizeof(a),0),"GPU start");check(dev.c->GetData(stop.Get(),&b,sizeof(b),0),"GPU stop");
        require(b>=a&&batch.frames.size()==unsigned(phases),"missing output");outputs+=batch.frames.size();
        const double wall=elapsed(begin);records.push_back({wall,double(b-a)*1000.0/frequency.Frequency});
        deadlineMisses+=wall>1000.0/fps;
        // Fixed input pacing includes ordinary idle time in board energy/frame.
        std::this_thread::sleep_until(epoch+std::chrono::nanoseconds(int64_t(i+1)*1000000000/fps));
    }
    const double intervalMs=elapsed(epoch);
    std::cout<<"DONE outputs="<<outputs<<" inferences="<<MotionSynthesizer::inference_count()-inferenceBefore
             <<" deadlineMisses="<<deadlineMisses<<" intervalMs="<<intervalMs<<std::endl;std::cin.get();
    std::cerr<<"HEAD cachedPairs="<<MotionSynthesizer::head_cache_pair_count()-headBefore<<" cachedPhases="<<MotionSynthesizer::head_cache_phase_count()-phaseBefore<<" planLoads="<<MotionSynthesizer::head_cache_load_count()<<'\n';
    std::ofstream csv(argv[7]);csv<<"pair,wall_ms,gpu_span_ms\n"<<std::setprecision(12);
    for(size_t i=0;i<records.size();++i)csv<<i<<','<<records[i].wall<<','<<records[i].gpu<<'\n';
    require(MotionSynthesizer::inference_count()-inferenceBefore==uint64_t(iterations)*phases,"inference count changed");
    timeEndPeriod(1);return 0;
}catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<std::endl;timeEndPeriod(1);return 1;}}
