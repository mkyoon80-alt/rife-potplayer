// Optional work during graph connection, never a condition for Run or first
// original delivery. Only CUDA/TRT objects are prepared; no D3D context commands.
std::atomic<uint64_t> preload_jobs{0};
struct PreloadRequest {
    std::filesystem::path directory;
    ComPtr<IDXGIAdapter> adapter;
    HMODULE module=nullptr;
};
std::mutex& preload_mutex(){static std::mutex value;return value;}
std::set<std::wstring>& preload_keys(){static std::set<std::wstring> value;return value;}
DWORD WINAPI preload_entry(void* argument){
    HMODULE module=nullptr;
    {
        std::unique_ptr<PreloadRequest> request(static_cast<PreloadRequest*>(argument));
        module=request->module;
        try{
            auto log=playback_diagnostics::acquire();
            auto note=[&](const char* event){playback_diagnostics::emit(log,0,0,event,[]{return "scope=package-preparation";});};
            note("preload-begin");int gpu=-1;
            auto shared=cached_adapter_engine(request->directory,request->adapter.Get(),gpu);
            note("preload-engine-ready");shared->prewarm_context();note("preload-context-ready");
        }
        catch(const std::exception& error){OutputDebugStringA((std::string("RIFE optional startup preparation: ")+error.what()+"\n").c_str());}
        catch(...){OutputDebugStringA("RIFE optional startup preparation failed; lazy initialization remains available\n");}
    }
    --preload_jobs;
    // Keep this DLL loaded until the thread has left its code, including when
    // the player abandons a graph while preparation is still in progress.
    FreeLibraryAndExitThread(module,0);
}
bool request_adapter_preload(const std::filesystem::path& directory,IDXGIAdapter* adapter)noexcept{
    if(!adapter||environment_flag(L"RIFE_DISABLE_STARTUP_PRELOAD"))return false;
    HMODULE module=nullptr;
    try{
        DXGI_ADAPTER_DESC desc{};ck(adapter->GetDesc(&desc),"Preload adapter identity");
        if(desc.VendorId!=0x10de)return false;
        auto dir=std::filesystem::absolute(directory).lexically_normal();
        const auto key=dir.wstring()+L"/"+std::to_wstring(desc.AdapterLuid.HighPart)+L"/"+std::to_wstring(desc.AdapterLuid.LowPart);
        std::lock_guard<std::mutex> lock(preload_mutex());
        if(preload_keys().count(key))return true;
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,reinterpret_cast<LPCWSTR>(&preload_entry),&module))return false;
        auto request=std::make_unique<PreloadRequest>();request->directory=std::move(dir);request->adapter=adapter;request->module=module;
        preload_keys().insert(key);++preload_jobs;
        HANDLE thread=CreateThread(nullptr,0,preload_entry,request.get(),0,nullptr);
        if(!thread){--preload_jobs;preload_keys().erase(key);FreeLibrary(module);return false;}
        request.release();module=nullptr;CloseHandle(thread);return true;
    }catch(...){if(module)FreeLibrary(module);return false;}
}
bool request_preload(const std::filesystem::path& directory,ID3D11Device* device)noexcept{
    if(!device)return false;
    try{ComPtr<IDXGIDevice> dxgi;ComPtr<IDXGIAdapter> adapter;
        ck(device->QueryInterface(IID_PPV_ARGS(&dxgi)),"Preload DXGI device");ck(dxgi->GetAdapter(&adapter),"Preload adapter");
        return request_adapter_preload(directory,adapter.Get());
    }catch(...){return false;}
}
