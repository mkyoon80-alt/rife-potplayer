#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include "nvof/phase_quality.hpp"
namespace nvof {
struct Frame {
    int width = 0;
    int height = 0;
    int64_t pts = 0; // DirectShow units: 100 ns; packed NV12, stride == width.
    std::vector<uint8_t> pixels;
};
class RifeEngine {
public:
    explicit RifeEngine(const std::filesystem::path& runtime_directory);
    ~RifeEngine();
    RifeEngine(const RifeEngine&) = delete;
    RifeEngine& operator=(const RifeEngine&) = delete;
    // Generate up to 32 strictly increasing timestamps inside an original pair.
    // Native D3D11 synthesis with upload/readback for system-memory transport.
    // Empty requests do no work. Hard cuts return no frames and reset history.
    // Output frames own their pixels; no input memory is retained by reference.
    PhaseBatch<Frame> interpolate_pair(const Frame& previous, const Frame& current,
                                      const std::vector<int64_t>& timestamps);
    void reset() noexcept;
    std::string device_name() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
