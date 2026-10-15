#pragma once
#include "gpu_engine.hpp"
#include "pipeline.hpp"
#include <functional>
#include <optional>
namespace nvof {
struct GpuOutputFrame { GpuFrame frame; int64_t stop = 0; bool discontinuity = false; };
using GpuEmit = std::function<bool(const GpuOutputFrame&)>;
}
