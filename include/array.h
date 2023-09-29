#pragma once

#include <array>

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