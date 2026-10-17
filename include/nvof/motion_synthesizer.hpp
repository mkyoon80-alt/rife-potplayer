#pragma once
#include <memory>
#include <array>
#include <filesystem>
#include <cstdint>
#include <functional>
#include <atomic>
#include <d3d11.h>
#include "rife_timing.hpp"
namespace nvof {
// Control-path cancellation, distinct from a failed GPU execution. The caller
// still owns all buffers until already submitted GPU work has been drained.
struct GpuWorkCancelled {};
// RIFE 4.25 Standard 960 / TensorRT FP16. D3D11 surfaces, CUDA interop.
// No builder or runtime shader compiler is linked into playback.
// Caller holds the shared immediate-context lock and restores pipeline state.
// prepare retains sources until the next call; outputs must not alias inputs.
struct MotionAnalysisInfo {
    unsigned width = 0, height = 0;
};
class MotionSynthesizer {
public:
    MotionSynthesizer(ID3D11Device*, ID3D11DeviceContext*, const std::filesystem::path& runtime_directory = {});
    ~MotionSynthesizer();
    void set_hdr_transfer(unsigned transfer) noexcept;
    // Called with the D3D device scope held. Collects ready queries without flushing.
    void begin_timing(bool enabled,int64_t pair_pts);
    const RifeTiming& timing()const noexcept;
    // Opportunistic preparation; never blocks playback start or first original.
    static bool preload_async(const std::filesystem::path&,ID3D11Device*) noexcept;
    // Connection-time decoder adapter selection; same keyed cache as the
    // device-based entry point. Invalid/non-NVIDIA adapters fail harmlessly.
    static bool preload_adapter_async(const std::filesystem::path&,unsigned adapter_index) noexcept;
    static uint64_t preload_pending() noexcept;
    static uint64_t optional_preload_pending() noexcept;
    static uint64_t execution_create_count() noexcept;
    static uint64_t engine_load_count() noexcept;
    static uint64_t inference_count() noexcept;
    static uint64_t graph_launch_count() noexcept;
    static uint64_t graph_build_count() noexcept;
    static uint64_t graph_fallback_count() noexcept;
    MotionSynthesizer(const MotionSynthesizer&) = delete;
    MotionSynthesizer& operator=(const MotionSynthesizer&) = delete;
    // Optional synchronous completion boundary. It may restore/release the
    // caller's shared context while waiting, but must reacquire it and restore
    // our exact pipeline state before returning (also on failure).
    using GpuWait = std::function<void()>;
    // Optional same-pair gate runs after input completion, before RIFE inference.
    // It may read earlier GPU statistics while the caller owns the context.
    // False leaves the pair unprepared; the caller must hold the original frame
    // and invalidate history. No earlier pair's decision may be reused.
    using PairReady = std::function<bool()>;
    // Optional trusted pool views in A-Y, B-Y, A-UV, B-UV order. They must
    // refer to the supplied single-slice textures with matching UNORM formats.
    // References are retained only for the prepared pair, never across reset.
    using CpuWork = std::function<void(const std::function<void()>&)>;
    bool initialization_pending() const noexcept;
    using SourcePlanes=std::array<ID3D11ShaderResourceView*,4>;
    void prepare(ID3D11Texture2D*, ID3D11Texture2D*, int width, int height, const GpuWait& wait = {}, const PairReady& ready = {}, const SourcePlanes& planes = {},const CpuWork& initialization = {});
    // True only when this source pair passed the scene/identity gate.
    bool prepared() const noexcept;
    // Resource preparation and synthetic CUDA warmup, after originals are
    // delivered. No source images, D3D draw/dispatch or unsatisfied queue Wait.
    // -1: preparation pending; otherwise number of synthetic enqueues.
    // Submission is not readiness: startup_ready checks actual GPU completion.
    int startup_warmup(int width=0,int height=0,bool multiple_outputs=false);
    bool startup_ready() const noexcept;
    HANDLE startup_warmup_event() const noexcept;
    MotionAnalysisInfo analysis_info() const noexcept;
    struct PhaseTarget { ID3D11RenderTargetView* y; ID3D11RenderTargetView* uv; float fraction; };
    // Bounded CUDA batch; outputs must stay alive until this call returns.
    using CompletionWait = std::function<void(HANDLE)>;
    // Optional CUDA-only setup may block on the background module loader. The
    // host releases/restores its shared D3D context while retaining ownership
    // of this engine and all source/output leases, including on failure.
    bool optional_setup_pending(size_t count) const noexcept;
    void render_phases(const PhaseTarget*,size_t count,const CompletionWait& completion_wait = {},const CpuWork& optional_setup = {},const std::atomic<bool>* cancelled = nullptr);
    static uint64_t handoff_count() noexcept;
    static uint64_t head_cache_pair_count() noexcept;
    static uint64_t head_cache_phase_count() noexcept;
    static uint64_t head_cache_load_count() noexcept;
    static uint64_t head_cache_context_count() noexcept;
    void render_phase(ID3D11RenderTargetView* y, ID3D11RenderTargetView* uv, float fraction);
    // Discard prepared image references while retaining compatible allocations.
    // Every new pair overwrites both guide inputs.
    void invalidate_history() noexcept;
    // Release the per-instance execution context/resources, including after an execution failure.
    void reset() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
