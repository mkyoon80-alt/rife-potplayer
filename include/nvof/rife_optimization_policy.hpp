#pragma once

namespace nvof {
// Source dimensions, before renderer scaling. The long edge includes cropped
// 3840x1604 films and their portrait equivalents without treating QHD as UHD.
constexpr bool rife_packed_motion_resolution(int width,int height) noexcept {
    return width>=2 && height>=2 && (width>=3840 || height>=3840);
}
}
