#pragma once

#include <array>
#include <execution>
#include <iostream>
#include <functional>  // for std::invoke

// Map an array of functions all over the same input
template <typename F, typename Tin, typename Tout, size_t N>
auto map_array(const std::array<F, N>& arr, const Tin& input) -> std::array<Tout, N> {
    std::array<Tout, N> output{};
    std::transform(std::execution::par_unseq, arr.begin(), arr.end(), output.begin(),
                   [&](const F& f) { return f(input); });
    return output;
}
