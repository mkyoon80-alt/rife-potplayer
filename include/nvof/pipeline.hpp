#pragma once
#include "engine.hpp"
#include <functional>
#include <optional>
namespace nvof {
struct Rate { int64_t num=60000; int64_t den=1001; };
struct OutputFrame { Frame frame; int64_t stop=0; bool discontinuity=false; };
using Emit = std::function<bool(const OutputFrame&)>;
}
