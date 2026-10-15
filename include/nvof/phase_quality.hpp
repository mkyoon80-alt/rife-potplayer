#pragma once
#include <cstdint>
#include <vector>
namespace nvof {
struct PairQuality {
    bool scene_cut = false;
    bool repetition_known = false;
    bool identical_warp_skipped = false;
    uint32_t repeated_mask = 0;
    uint32_t native_synthesized_mask = 0; // Phases composed from RIFE flow and mask.
};
template<class FrameType> struct PhaseBatch {
    std::vector<FrameType> frames;
    PairQuality quality;
};
}
