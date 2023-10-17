#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <execution>
#include <type_traits>

template <template <typename, typename...> class C, typename T, typename... Args>
//requires requires (C<T, Args...> c) { c.begin(); c.end(); }
T mac(const C<T, Args...>& a, const C<T, Args...>& b, T init = T{}) {
    //static_assert(std::is_arithmetic_v<T>, "Container must hold an arithmetic type");
    if (a.size() != b.size()) {
        throw std::invalid_argument("Both containers must have the same size.");
    }

    return std::transform_reduce(
        std::execution::par_unseq, // Use parallel and vectorized execution
        a.begin(), a.end(), // Range of first vector
        b.begin(), // Range of second vector
        init, //static_cast<T>(0), // Initial value
        std::plus<>(), // Accumulate
        std::multiplies<>() // Multiply
    );
}

// Specialization for std::array
template <typename T, std::size_t N>
T mac(const std::array<T, N>& a, const std::array<T, N>& b, T init = T{}) {
    //static_assert(std::is_arithmetic_v<T>, "Container must hold an arithmetic type");

    return std::transform_reduce(
        std::execution::par_unseq, // Use parallel and vectorized execution
        a.begin(), a.end(), // Range of first vector
        b.begin(), // Range of second vector
        init, //static_cast<T>(0), // Initial value
        std::plus<>(), // Accumulate
        std::multiplies<>() // Multiply
    );
}
