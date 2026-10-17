#pragma once
#include <cstdint>
#include <array>
#include "phase_quality.hpp"
#include "hdr_metadata.hpp"
#include "motion_synthesizer.hpp"
#include <vector>
#include <filesystem>
#include <memory>
#include <string>
#include <d3d11.h>
#include <wrl/client.h>
namespace nvof {
struct GpuFrame {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    UINT array_slice = 0;
    int width = 0;
    int height = 0;
    int64_t pts = 0;
    std::shared_ptr<void> lease; // Keeps a pooled surface reserved until every owner releases it.
    std::shared_ptr<const HdrMetadata> hdr; // Immutable per-frame snapshot, never global history.
    unsigned hdr_transfer = 0; // PQ/HLG select BT.2020 guide conversion; output stays native.
};
enum class GpuCompletionMode { blocking, context_ordered };
// Wall-clock API/lock/wait measurements, not GPU timestamp durations.
struct GpuPairTiming {
    std::shared_ptr<const RifeTiming> rife;
    double context_wait_ms=0,scene_ms=0,prepare_ms=0,submit_ms=0,completion_ms=0,shared_wait_ms=0;
    // Subintervals of shared_wait_ms. Query fallback completion also includes
    // protected-context polling; neither path measures GPU execution alone.
    double wait_submit_ms=0,wait_completion_ms=0,context_resume_ms=0;
    // Subset of shared wait: CPU waits while earlier D3D work may proceed.
    double cuda_ready_wait_ms=0,cuda_ready_resume_ms=0;
    uint32_t cuda_ready_wait_calls=0;
    double event_arm_ms=0,event_sleep_ms=0;
    double decoder_lock_ms=0,multithread_lock_ms=0,state_swap_ms=0;
    double wait_service_ms=0;
    double target_allocation_ms=0;
    uint32_t wait_service_calls=0;
    uint32_t shared_wait_calls=0,combined_scene_pairs=0;
    uint32_t event_wait_calls=0,query_wait_calls=0;
    uint32_t flush_event_wait_calls=0; // Subset of event_wait_calls; Flush1 ALL.
    uint32_t queued_output_flushes=0;
    std::array<double,3> wait_each_ms{},resume_each_ms{};

};
// NV12/P010 GPU-only image path. Native Y/UV processing preserves stored
// channel values; the caller remains responsible for supported color metadata.
// Input textures must remain unchanged until
// the method returns. For decoder arrays, retain IMediaSample through copy().
// Returned textures are engine-owned snapshots. The default waits for GPU
// completion. context_ordered instead submits all work to this device's shared
// immediate context before returning: consumers MUST submit their GPU use to
// that same context. D3D command/resource ordering then protects source reuse;
// consumers of other GPU APIs must establish their own completion dependency.
class GpuRifeEngine {
public:
    GpuRifeEngine(const std::filesystem::path& runtime_directory,
                  ID3D11Device* device, ID3D11DeviceContext* context,
                  HANDLE borrowed_decoder_mutex = nullptr,
                  GpuCompletionMode completion = GpuCompletionMode::blocking);
    ~GpuRifeEngine();
    GpuRifeEngine(const GpuRifeEngine&) = delete;
    GpuRifeEngine& operator=(const GpuRifeEngine&) = delete;
    // Preserves NV12 or P010 in owned snapshots (P010 low six bits are cleared).
    // Both formats honor the selected completion contract. Native D3D11 plane
    // draws/copies use the same context ordering; no CUDA interop is involved.
    // No range, transfer-function, gamut conversion or tone mapping is performed.
    // Boundary originals may be delivered before any later interpolation
    // submission. submit_now flushes their commands without a completion wait.
    GpuFrame copy(const GpuFrame& input, bool submit_now = false);
    PhaseBatch<GpuFrame> interpolate_pair(const GpuFrame& previous, const GpuFrame& current, const std::vector<int64_t>& timestamps,const std::atomic<bool>* cancelled = nullptr);
    void reset() noexcept;
    std::string device_name() const;
    bool queued_completion() const;
    MotionAnalysisInfo analysis_info() const;
    void enable_timing(bool enabled,bool rife_detail=false,unsigned profile_interval=1);
    GpuPairTiming pair_timing() const;
    // Runs only with the shared device scope released, on the calling thread.
    // Return an optional borrowed wake event for ready-output delivery. The
    // callback must never re-enter this engine, wait for Run, or throw. Its
    // failure is reported by the caller after GPU completion, before new output.
    using WaitService = std::function<HANDLE()>;
    void set_wait_service(WaitService service);
    int startup_warmup(int width=0,int height=0,bool multiple_outputs=false);
    bool startup_ready() const;
    // Quiesced engines only. Retain the host's borrowed-mutex owner until GPU
    // cleanup completes; never let optional module loading block graph Stop.
    // False leaves ownership with the caller for ordinary synchronous cleanup.
    static bool retire_during_preparation(std::unique_ptr<GpuRifeEngine>&,std::shared_ptr<void> host_lifetime) noexcept;
    static unsigned retirement_pending() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
