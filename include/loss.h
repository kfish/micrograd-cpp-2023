#pragma once

#include <array>
#include <cmath>
#include <numeric>

namespace ai {

// Template function for double
template <typename T>
double mse_loss(const T& predicted, const T& ground_truth) {
    static_assert(std::is_arithmetic<T>::value, "Type must be arithmetic");
    return std::pow(predicted - ground_truth, 2);
}

// Overloaded function for Value<double>
template <typename T>
Value<T> mse_loss(const Value<T>& predicted, const T& ground_truth) {
    static_assert(std::is_arithmetic<T>::value, "Type must be arithmetic");
    return pow(predicted - ground_truth, 2);
}

template <typename T>
Value<T> mse_loss(const Value<T>& predicted, const Value<T>& ground_truth) {
    static_assert(std::is_arithmetic<T>::value, "Type must be arithmetic");
    return pow(predicted - ground_truth, 2);
}

// Wrapper function for containers
template <template <typename...> class Container, typename T>
double mse_loss(const Container<T>& predicted, const Container<T>& ground_truth) {
    static_assert(std::is_arithmetic<T>::value || std::is_same<Value<T>, T>::data_, "Type must be arithmetic or Value<T>");

    if(predicted.size() != ground_truth.size()) {
        throw std::invalid_argument("Size mismatch between predicted and ground truth data");
    }

    double loss = 0.0;
    auto truth_it = ground_truth.begin();

    for (const auto& pred : predicted) {
        loss += mse_loss(pred, *truth_it);
        ++truth_it;
    }

    return loss / predicted.size();
}

template<typename T, std::size_t N>
T mse_loss(const std::array<T, N>& predictions, const std::array<T, N>& ground_truth) {
    T sum_squared_error = std::inner_product(predictions.begin(), predictions.end(), ground_truth.begin(), T(0),
        std::plus<>(),
        [](T pred, T truth) { return std::pow(pred - truth, 2); }
    );
    return sum_squared_error / static_cast<T>(N);
}

template<typename T, std::size_t N>
Value<T> mse_loss(const std::array<Value<T>, N>& predictions, const std::array<T, N>& ground_truth) {
    Value<T> sum_squared_error = std::inner_product(predictions.begin(), predictions.end(), ground_truth.begin(), leaf<T>(0),
        std::plus<>(),
        [](Value<T> pred, T truth) { return pow(pred - truth, 2); }
    );
    return sum_squared_error / leaf<T>(N);
}

};
