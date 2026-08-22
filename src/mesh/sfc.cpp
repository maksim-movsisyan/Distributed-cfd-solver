#include "cfd/mesh/sfc.hpp"

#include <algorithm>
#include <cmath>

static uint32_t spread_bits(uint32_t v) {
    v &= (1u << kSFCBits) - 1;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

uint64_t sfc_key(double fx, double fy, double fz) {
    auto q = [](double f) -> uint32_t {
        if (!(f > 0.0)) return 0;  // nan and <= 0
        if (f >= 1.0) return (1u << kSFCBits) - 1;
        return static_cast<uint32_t>(std::lround(f * ((1u << kSFCBits) - 1)));
    };
    const uint64_t x = spread_bits(q(fx));
    const uint64_t y = spread_bits(q(fy));
    const uint64_t z = spread_bits(q(fz));
    return x | (y << 1) | (z << 2);
}
