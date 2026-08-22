#pragma once
// Space-filling curve keys for spatial data locality.
//
// Stage 1 uses the Morton (z-order) code: trivially correct and giving
// locality comparable to Hilbert up to a constant. Upgrading to Hilbert
// is a separate task; the sfc_key() interface will not change.

#include <cstdint>

// Key from normalized [0,1]^3 coordinates (21 bits per axis, 63 bits total).
uint64_t sfc_key(double x /*[0,1]*/, double y, double z);

// Bits per coordinate in the key.
constexpr int kSFCBits = 21;
