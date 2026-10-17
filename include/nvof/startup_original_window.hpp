#pragma once
#include <cstdint>
#include <limits>
#include <optional>

namespace nvof {
// Owner serializes graph-control and streaming access. This policy never waits
// or owns GPU resources. The caller may wait only after originals covering the
// initial media interval have already been delivered.
class StartupOriginalWindow {
public:
    struct Decision { bool original=false; uint64_t wait_ms=0; };
    void reset() noexcept { active_=true; started_=false; preparing_=false; first_pts_.reset(); }
    void cancel() noexcept { active_=false; preparing_=false; }
    bool active() const noexcept { return active_; }
    bool started() const noexcept { return started_; }
    bool originals_cover(int64_t stop) const noexcept {
        return first_pts_ && stop>=*first_pts_ && uint64_t(stop)-uint64_t(*first_pts_)>=uint64_t(media_span_);
    }
    // Initial preroll only, bounded by the same media interval as originals
    // after Run. A later Pause or seek never permits extra output.
    bool preroll_original(int64_t pts) const noexcept {
        return active_ && !started_ && first_pts_ && pts>=*first_pts_ &&
            uint64_t(pts)-uint64_t(*first_pts_)<uint64_t(media_span_);
    }
    void set_rate(double rate) noexcept {
        const double span=1000000.0*rate;
        const auto maximum=(std::numeric_limits<int64_t>::max)();
        media_span_=span>=double(maximum)?maximum:span<1?1:int64_t(span);
    }
    bool run(uint64_t now) noexcept {
        if(!active_ || started_)return false;
        started_=true;begin_ms_=now;
        const auto maximum=(std::numeric_limits<uint64_t>::max)();
        deadline_ms_=now>maximum-100?maximum:now+100;
        return true;
    }
    uint64_t elapsed(uint64_t now) const noexcept { return started_ && now>=begin_ms_?now-begin_ms_:0; }
    // Only the boundary pair may be prepared early. Prepared timestamped
    // output may be queued immediately; the original window is not extended.
    bool begin_preparation(uint64_t now,int64_t pts) noexcept {
        if(!decide(now,pts).wait_ms)return false;
        preparing_=true;return true;
    }
    // Remaining preparation budget, not a required delay before delivery.
    uint64_t output_wait_ms(uint64_t now) noexcept {
        if(!active_ || !preparing_ || !started_)return 0;
        if(now>=deadline_ms_){active_=false;preparing_=false;return 0;}
        return deadline_ms_-now;
    }
    Decision decide(uint64_t now,int64_t pts) noexcept {
        if(!active_ || pts<0)return {};
        if(!first_pts_)first_pts_=pts;
        if(pts<*first_pts_) { active_=false;return {}; }
        if(!started_)return {true,0};
        if(preparing_)return {};
        // Media coverage and preparation time are distinct. A late decoder or
        // module loader must not shorten the original/audio runway to one
        // frame merely because 100 ms elapsed since this filter's Run call.
        if(uint64_t(pts)-uint64_t(*first_pts_)<uint64_t(media_span_))return {true,0};
        if(now>=deadline_ms_) { active_=false;return {}; }
        // Decode-ahead must not turn a 100-ms interval into an allocator full
        // of long-duration originals. Resume inference at the wall deadline.
        if(uint64_t(pts)-uint64_t(*first_pts_)>=uint64_t(media_span_))return {false,deadline_ms_-now};
        return {true,0};
    }
private:
    bool active_=false,started_=false,preparing_=false;
    uint64_t begin_ms_=0,deadline_ms_=0;
    int64_t media_span_=1000000;
    std::optional<int64_t> first_pts_;
};
}
