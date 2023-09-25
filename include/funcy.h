#pragma once

#include <array>
#include <execution>
#include <iostream>
#include <functional>  // for std::invoke

template<typename T, typename Arg, std::size_t N>
std::array<T, N> map_array(const std::array<std::function<T(Arg)>, N>& input, const Arg& arg) {
    std::array<T, N> output;
    std::transform(std::execution::par, input.begin(), input.end(), output.begin(),
                   [&arg](const auto& func) { return std::invoke(func, arg); });
    return output;
}

