#pragma once

namespace cfd::solver::time {

struct SteadyMode {
    static constexpr bool kIsUnsteady = false;
    static constexpr int kBdfOrder = 0;
    static constexpr const char* name() noexcept { return "Steady"; }
};

template <int Order = 2>
struct DualTimeMode {
    static_assert(Order == 1 || Order == 2, "DTS supports BDF1 and BDF2 only");
    static constexpr bool kIsUnsteady = true;
    static constexpr int kBdfOrder = Order;
    static constexpr const char* name() noexcept { return Order == 1 ? "DTS(BDF1)" : "DTS(BDF2)"; }
};

} // namespace cfd::solver::time