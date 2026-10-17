#pragma once
#include <array>
#include <cstdint>
namespace nvof {
// CPU wall intervals include driver/queue waits. CUDA and D3D intervals are
// device clock spans, not utilization. CPU and GPU columns overlap: do not add.
struct RifePhaseTiming {
    uint64_t pair=0;int64_t pts=0;float phase=0;unsigned width=0,height=0;
    bool external_fence=false;
    bool head_cached=false,head_cache_first=false,packed_layout=false;
    bool cuda_valid=false,cuda_pending=false,graph_replayed=false,graph_created=false;
    unsigned batch_size=1,map_count=0,unmap_count=0,copy_count=0;
    double copy_cpu_ms=0,copy_gpu_ms=0;
    double status_query_cpu_ms=0;
    double time_submit_cpu_ms=0,device_cpu_ms=0,map_cpu_ms=0,bind_cpu_ms=0;
    double enqueue_cpu_ms=0,unmap_cpu_ms=0,sync_cpu_ms=0,composite_submit_cpu_ms=0;
    double event_read_cpu_ms=0,total_cpu_ms=0;
    // Map/unmap event spans include stream handoff and host submission gaps;
    // they are NOT bytes-copied timings. Inference excludes these boundaries.
    double map_stream_ms=0,inference_gpu_ms=0,unmap_stream_ms=0;
};
enum class RifeGpuKind : unsigned { guide,time,composite };
struct RifeGpuTiming {
    uint64_t pair=0;int64_t pts=0;float phase=0;unsigned width=0,height=0;
    RifeGpuKind kind=RifeGpuKind::guide;double gpu_ms=0;
};
struct RifeTiming {
    bool enabled=false;uint64_t pair=0;int64_t pts=0;
    double initialize_cpu_ms=0,guide_submit_cpu_ms=0,query_collect_cpu_ms=0;
    double init_engine_ms=0,init_context_ms=0,init_buffers_ms=0,init_shaders_ms=0;
    double init_packed_ms=0,head_prepare_cpu_ms=0;
    double init_take_context_ms=0,init_stream_ms=0;
    std::array<RifePhaseTiming,32> phases{};unsigned phase_count=0;
    std::array<RifePhaseTiming,128> completed_phases{};unsigned completed_phase_count=0;
    // Completed D3D samples can belong to earlier pairs. Always use their IDs.
    std::array<RifeGpuTiming,128> gpu{};unsigned gpu_count=0;
    unsigned cuda_invalid=0,cuda_skipped=0,d3d_invalid=0,d3d_skipped=0;
};
}
