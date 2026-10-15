#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
namespace nvof {
// The playback thread never waits for the writer or touches the log file.
// Bounded queue, try-lock, dropped messages reported rather than backpressure.
class PerformanceLog {
public:
    using Batch=std::deque<std::string>;
    using Sink=std::function<void(const Batch&,uint64_t)>;
private:
    std::mutex mutex_;std::condition_variable ready_;Batch pending_;
    std::atomic<uint64_t> dropped_{0};bool stopping_=false;
    Sink sink_;std::thread worker_;
    void run()noexcept{
        try { for(;;){Batch batch;bool stopping;
            {std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock,[&]{return stopping_||!pending_.empty();});
                if(!stopping_)ready_.wait_for(lock,std::chrono::seconds(1),[&]{return stopping_||pending_.size()>=128;});
                batch.swap(pending_);stopping=stopping_;
            }
            try{if(!batch.empty())sink_(batch,dropped_.exchange(0));}catch(...){}
            if(stopping)return;
        } } catch (...) { /* Diagnostics can stop without affecting playback. */ }
    }
public:
    explicit PerformanceLog(Sink sink):sink_(std::move(sink)),worker_([this]{run();}){}
    ~PerformanceLog(){
        {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}
        ready_.notify_one();if(worker_.joinable())worker_.join();
    }
    bool submit(std::string message)noexcept{
        try{
            std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
            if(!lock.owns_lock()||stopping_||pending_.size()>=256){++dropped_;return false;}
            pending_.push_back(std::move(message));
            // Wake only for the first message or a full batch, not every line.
            // Repeated wakeups would contend with the playback try-lock.
            const bool wake=pending_.size()==1||pending_.size()==128;
            lock.unlock();if(wake)ready_.notify_one();return true;
        }catch(...){++dropped_;return false;}
    }
};
}
