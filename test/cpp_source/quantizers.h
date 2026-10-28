#pragma once
#include "ac_types/ac_fixed.h"
#include "ac_types/ac_std_float.h"
#include <cmath>
#include <cstddef>

namespace qtest {
template <typename T> void quantize(const float *in, float *out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = T(in[i]).to_double();
}

// no inf/nan, matching behavior using one extra exponent with explicit clamping
template <int M, int E, int E0> void minifloat(const float *in, float *out, size_t n) {
    constexpr int e_high = (1 << (E - 1)) - 1 + E0;
    const float max = std::ldexp(2.0f - std::ldexp(1.0f, -M), e_high);
    for (size_t i = 0; i < n; ++i) {
        float q = std::ldexp(ac_std_float<2 + E + M, E + 1>(std::ldexp(in[i], -e_high)).to_float(), e_high);
        out[i] = std::fmin(std::fmax(q, -max), max);
    }
}

inline void binary(const float *in, float *out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = in[i] >= 0 ? 1.0f : -1.0f;
}
} // namespace qtest
