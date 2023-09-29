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

template <typename Func, typename T, std::size_t N>
auto array_transform(Func&& func, const std::array<T, N>& input) {
    std::array<decltype(func(std::declval<T>())), N> output;
    std::transform(input.begin(), input.end(), output.begin(), std::forward<Func>(func));
    return output;
}

template <template <typename...> class Container, typename Func, typename T>
auto map_container(const Container<T>& input, Func&& func) {
    Container<T> output;
    output.resize(input.size());
    std::transform(input.begin(), input.end(), output.begin(), std::forward<Func>(func));
    return output;
}
