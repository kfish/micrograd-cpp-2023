#pragma once

#include <random>
#include <array>
#include <iostream>
#include <type_traits>

// Static inline function to generate a random T
template <typename T>
static inline T randomValue() {
    static unsigned int seed = 42;
    static thread_local std::mt19937 gen(seed++);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    return dist(gen);
}

// Static inline function to generate a random std::array<T, N>
template <typename T, std::size_t N>
static inline std::array<T, N> randomArray() {
    std::array<T, N> arr;
    for (auto& elem : arr) {
        elem = randomValue<T>();
    }
    return arr;
}

template<typename T, std::size_t N>
class PrettyArray : public std::array<T, N> {
public:
    PrettyArray() = default;
    PrettyArray(const std::array<T, N>& arr) : std::array<T, N>(arr) {}

    friend std::ostream& operator<<(std::ostream& os, const PrettyArray& arr) {
        os << "[ ";
        for (const auto& elem : arr) {
            os << elem << ' ';
        }
        os << ']';
        return os;
    }
};
