#include "nvof/performance_log.hpp"
#include <future>
#include <iostream>
#include <stdexcept>
int main(){try{
    std::promise<void> entered,release;auto gate=release.get_future().share();auto started=entered.get_future();
    std::atomic<unsigned> batches{0};std::atomic<uint64_t> drops{0};
    auto log=std::make_unique<nvof::PerformanceLog>([&](const nvof::PerformanceLog::Batch&,uint64_t dropped){drops+=dropped;if(batches++==0){entered.set_value();gate.wait();}});
    for(int i=0;i<128;++i)log->submit("first batch");
    if(started.wait_for(std::chrono::seconds(3))!=std::future_status::ready)throw std::runtime_error("Writer did not start");
    // The sink is deliberately blocked. Playback must fill its bounded queue
    // and drop excess diagnostics without depending on sink completion.
    auto push=std::async(std::launch::async,[&]{unsigned accepted=0;for(int i=0;i<10000;++i)accepted+=log->submit("bounded message");return accepted;});
    bool ready=push.wait_for(std::chrono::milliseconds(500))==std::future_status::ready;
    release.set_value();unsigned accepted=push.get();log.reset();
    if(!ready||accepted>256||!drops)throw std::runtime_error("Logging blocked playback or queue is unbounded");
    std::cout<<"PASS blocked writer did not block producer; bounded queue and drop accounting\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
