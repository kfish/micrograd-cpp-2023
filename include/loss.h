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

template<typename T, size_t N>
T mse_loss(const std::array<T, N>& predictions, const std::array<T, N>& ground_truth) {
    T sum_squared_error = std::inner_product(predictions.begin(), predictions.end(), ground_truth.begin(), T(0),
        std::plus<>(),
        [](T pred, T truth) { return std::pow(pred - truth, 2); }
    );
    return sum_squared_error / static_cast<T>(N);
}

template<typename T, size_t N>
Value<T> mse_loss(const std::array<Value<T>, N>& predictions, const std::array<T, N>& ground_truth) {
    Value<T> sum_squared_error = std::inner_product(predictions.begin(), predictions.end(), ground_truth.begin(), leaf<T>(0),
        std::plus<>(),
        [](Value<T> pred, T truth) { return pow(pred - truth, 2); }
    );
    return sum_squared_error / leaf<T>(N);
}

template<typename T, size_t N, typename Arg>
class MSELoss {
    public:
        MSELoss(const std::function<Value<T>(const Arg&)>& func)
            : func_(func)
        {
            for (size_t i = 0; i < N; ++i) {
                predictions_[i] = RawValue<T>::make(0.0);
            }
            //std::cerr << "Cnstrct " << this << "=&MSELoss(func_=" << &func_
            //    << " from func=" << &func
            //    << std::endl;
           //std::array<double, 2> a = {2.0, -3.0};
           //auto x = func(a);
           //std::cerr << "Calculated x=" << x << std::endl;
        }

        //Value<T> operator()(const std::array<Value<T>, N>& input, const std::array<T, N>& ground_truth) {
        Value<T> operator()(std::array<Arg, N>& input, const std::array<T, N>& ground_truth) {
#if 0
            std::transform(input.begin(), input.end(), predictions_.begin(), func_);
            //return mse_loss(predictions_, ground_truth);
            return leaf(0.0);
#else
            //std::cerr << "CALC " << this << "=&MSELoss(func_=" << &func_ << std::endl;
            std::cerr << "Predictions: ";
            for (size_t i = 0; i < N; ++i) {
                //std::cerr << "Calculate " << i << "th prediction, using input value " << PrettyArray<T, N>(input[i]) << std::endl;
                //std::cerr << "Calculate " << i << "th prediction, using input value " << input[i][0] << "," << input[i][1] << std::endl;
                predictions_[i] = func_(input[i]);
                std::cerr << predictions_[i]->data() << " ";
            }
            std::cerr << '\n';
#endif
            return mse_loss(predictions_, ground_truth);
        }

    private:
        //const std::function<Value<T>(const std::array<Value<T>, N>&)>& func_;
        const std::function<Value<T>(const Arg&)> func_;
        std::array<Value<T>, N> predictions_;
};

#if 0
template <typename T, size_t N, size_t Nout>
class Extract1 {
    public:
        Extract1(const std::function<std::array<std::array<Value<T>, Nout>, N>(const std::array<Value<T>, N>&) func)
            : func_(func)
        {
        }

        std::array<Value<T>, 
};
#endif

};
