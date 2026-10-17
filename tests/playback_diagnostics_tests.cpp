#include "nvof/playback_diagnostics.hpp"
#include <future>
#include <iostream>
#include <stdexcept>
using namespace nvof;
using namespace std::chrono_literals;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::string read(const std::filesystem::path& path){std::ifstream file(path);return {std::istreambuf_iterator<char>(file),{}};}
int main(){try {
    wchar_t temporary[MAX_PATH]{};require(GetTempPathW(MAX_PATH,temporary)>0,"temp path");
    const auto folder=std::filesystem::path(temporary)/(L"rife-diagnostics-"+std::to_wstring(GetCurrentProcessId()));
    const auto fault=playback_diagnostics::line(7,2,"error","removedReason=0x887A0006\nerror=test");
    require(fault.find('\n')==std::string::npos && fault.find("instance=7 stream=2 event=error ")!=std::string::npos,"correlation or line escaping");
    require(playback_diagnostics::hex_code(E_FAIL)=="0x80004005","HRESULT encoding");
    playback_diagnostics::save(folder,{fault},0,1);
    playback_diagnostics::save(folder,{playback_diagnostics::line(8,1,"stream-start-ready","normal")},3,1);
    require(read(folder/L"playback.previous.log")==fault+"\n","rotation lost prior failure");
    require(read(folder/L"playback.last-error.log")==fault+"\n","reopen overwrote last failure");
    require(read(folder/L"playback.log").find("diagnosticDropped=3")!=std::string::npos,"drops hidden");
    std::promise<void> entered,release;auto gate=release.get_future().share();auto started=entered.get_future();
    std::atomic<unsigned> batches{0};std::atomic<uint64_t> drops{0};
    auto writer=std::make_shared<PerformanceLog>([&](const PerformanceLog::Batch&,uint64_t dropped){
        drops+=dropped;if(batches++==0){entered.set_value();gate.wait();}
    });
    playback_diagnostics::emit(writer,1,1,"stream-start-ready",[]{return "test";});
    require(started.wait_for(3s)==std::future_status::ready,"background writer did not start");
    auto producer=std::async(std::launch::async,[&]{
        for(int i=0;i<10000;++i)playback_diagnostics::emit(writer,1,1,"pause-begin",[]{return "test";});
        playback_diagnostics::emit(writer,1,1,"error",[]()->std::string{throw std::bad_alloc();});
    });
    const bool nonblocking=producer.wait_for(1s)==std::future_status::ready;
    release.set_value();producer.get();writer.reset();
    require(nonblocking && drops>0,"blocked disk delayed producer or queue not bounded");
    unsigned calls=0;
    auto failing=std::make_shared<PerformanceLog>([&](const PerformanceLog::Batch&,uint64_t){++calls;throw std::runtime_error("disk unavailable");});
    playback_diagnostics::emit(failing,1,1,"error",[]{return "test";});failing.reset();
    require(calls==1,"sink failure test did not run");
    for(const wchar_t* name:{L"playback.log",L"playback.previous.log",L"playback.last-error.log"})std::filesystem::remove(folder/name);
    std::filesystem::remove(folder);
    std::cout<<"PASS bounded/nonblocking logging, allocation/disk failure isolation, correlation, rotation and last-error retention\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
