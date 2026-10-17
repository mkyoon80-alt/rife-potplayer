#define wmain unused_adapter_probe_main
#include "rife_probe.cpp"
#undef wmain

int wmain(int argc,wchar_t** argv){try{
    require(argc==2,"rife_preload_adapter_tests runtime");
    Device dev;ComPtr<IDXGIDevice> dxgi;ComPtr<IDXGIAdapter> actual;
    check(dev.d.As(&dxgi),"DXGI device");check(dxgi->GetAdapter(&actual),"actual adapter");
    DXGI_ADAPTER_DESC actualDesc{};check(actual->GetDesc(&actualDesc),"actual LUID");
    ComPtr<IDXGIFactory1> factory;check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"factory");
    UINT selected=UINT_MAX;
    for(UINT i=0;;++i){ComPtr<IDXGIAdapter1> adapter;auto hr=factory->EnumAdapters1(i,&adapter);
        if(hr==DXGI_ERROR_NOT_FOUND)break;check(hr,"enumerate");DXGI_ADAPTER_DESC1 desc{};check(adapter->GetDesc1(&desc),"adapter LUID");
        if(desc.AdapterLuid.HighPart==actualDesc.AdapterLuid.HighPart&&desc.AdapterLuid.LowPart==actualDesc.AdapterLuid.LowPart)selected=i;
    }
    require(selected!=UINT_MAX,"selected adapter not found");
    require(!MotionSynthesizer::preload_adapter_async(argv[1],UINT_MAX),"invalid adapter accepted");
    const auto tick=Clock::now();
    require(MotionSynthesizer::preload_adapter_async(argv[1],selected),"adapter preload failed");
    require(MotionSynthesizer::preload_async(argv[1],dev.d.Get()),"same-device preload failed");
    const auto requestMs=elapsed(tick);
    auto pixels=pattern(320,180,0,true);auto frame=dev.upload(pixels,320,180,0,true);
    auto e=engine(dev,argv[1]);auto copy=e->copy(frame,true);
    require(dev.read(copy)==pixels,"first original changed while loading");
    const auto deadline=GetTickCount64()+30000;
    while(MotionSynthesizer::preload_pending()&&GetTickCount64()<deadline)Sleep(1);
    require(!MotionSynthesizer::preload_pending(),"preload did not finish");
    require(MotionSynthesizer::execution_create_count()==1,"adapter/device entry points duplicated contexts");
    require(e->startup_warmup()==2,"warmup failed after preload");
    require(MotionSynthesizer::execution_create_count()==1,"warmup ignored prepared context");
    e.reset();
    auto next=engine(dev,argv[1]);require(next->startup_warmup()==2,"cancel/reopen warmup failed");next.reset();
    require(MotionSynthesizer::execution_create_count()==1,"cancel/reopen lost reusable context");
    std::cout<<"PASS adapter="<<selected<<" requestMs="<<requestMs<<" firstOriginalExact=1 contexts=1 cancelAndReopen=1\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
