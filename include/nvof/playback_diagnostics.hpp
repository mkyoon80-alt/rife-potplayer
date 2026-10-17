#pragma once
#include "performance_log.hpp"
#include <windows.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace nvof::playback_diagnostics {
// Sparse lifecycle/error records only. Never call from a per-frame loop.
// Uses the existing bounded, nonblocking producer; all disk work is on its worker.
inline std::string hex_code(HRESULT value) {
    char text[16]{};sprintf_s(text,"0x%08lX",static_cast<unsigned long>(value));return text;
}
inline std::string line(unsigned instance,unsigned stream,const char* event,const std::string& detail) {
    SYSTEMTIME now{};GetLocalTime(&now);
    char prefix[192]{};
    sprintf_s(prefix,"%04u-%02u-%02u %02u:%02u:%02u.%03u pid=%lu tid=%lu tickMs=%llu instance=%u stream=%u event=%s ",
        now.wYear,now.wMonth,now.wDay,now.wHour,now.wMinute,now.wSecond,now.wMilliseconds,
        GetCurrentProcessId(),GetCurrentThreadId(),GetTickCount64(),instance,stream,event);
    std::string result=prefix;
    result.append(detail,0,8192);
    for(char& c:result)if(c=='\r'||c=='\n')c=' ';
    return result;
}
inline void save(const std::filesystem::path& directory,const PerformanceLog::Batch& batch,uint64_t dropped,
                 uintmax_t limit=2*1024*1024) {
    std::error_code error;std::filesystem::create_directories(directory,error);
    const auto path=directory/L"playback.log";
    if(std::filesystem::exists(path,error)&&std::filesystem::file_size(path,error)>limit) {
        const auto previous=directory/L"playback.previous.log";
        std::filesystem::remove(previous,error);error.clear();std::filesystem::rename(path,previous,error);
        if(error){std::ofstream truncate(path,std::ios::trunc);}
    }
    std::ofstream file(path,std::ios::app);
    if(dropped)file<<"diagnosticDropped="<<dropped<<'\n';
    for(const auto& text:batch) {
        file<<text<<'\n';
        // Keep the most recent failure across normal reopen/stop and rotation.
        if(text.find(" event=error ")!=std::string::npos) {
            std::ofstream last(directory/L"playback.last-error.log",std::ios::trunc);last<<text<<'\n';
        }
    }
}
inline std::shared_ptr<PerformanceLog> acquire() {
    static std::mutex mutex;static std::weak_ptr<PerformanceLog> weak;
    std::lock_guard<std::mutex> lock(mutex);if(auto existing=weak.lock())return existing;
    wchar_t local[32768]{};
    const DWORD length=GetEnvironmentVariableW(L"LOCALAPPDATA",local,_countof(local));
    if(!length||length>=_countof(local))return {};
    const auto directory=std::filesystem::path(local)/L"RifeStandardPotPlayer";
    auto service=std::make_shared<PerformanceLog>([directory](const PerformanceLog::Batch& batch,uint64_t dropped){save(directory,batch,dropped);});
    weak=service;return service;
}
template<class Details>
inline void emit(const std::shared_ptr<PerformanceLog>& writer,unsigned instance,unsigned stream,const char* event,Details&& details) noexcept {
    try {if(writer)writer->submit(line(instance,stream,event,details()));}
    catch(...) { /* Allocation/query/logging failures cannot change playback. */ }
}
}
