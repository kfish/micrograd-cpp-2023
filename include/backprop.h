#pragma once

#include <concepts>
#include <functional>
#include <iostream>
#include <fstream>

#include "loss.h"

namespace ai {

template <typename F, typename T, typename Arg>
concept CanBackProp = requires(F f, Arg arg, T learning_rate) {
    { f(arg) } -> std::convertible_to<Value<T>>;
    { f.adjust(learning_rate) } -> std::convertible_to<void>;
};


template<typename T, size_t N, typename Arg, typename F>
class BackPropImpl {
    public:
        BackPropImpl(const F& func, const std::string& loss_path)
            : func_(func), loss_output_(loss_path)
        {
        }

        MSELoss<T, N, Arg> loss_function() const {
            return MSELoss<T, N, Arg>(func_);
        }

        T operator()(std::array<Arg, N>& input, const std::array<T, N>& ground_truth,
                T learning_rate, int iterations)
        {
            auto loss_f = loss_function();
            T result;

            for (int i=0; i < iterations; ++i) {
                Value<double> loss = loss_f(input, ground_truth);

                result = loss->data();
                std::cerr << "Loss (" << iter_ << "):\t" << result << std::endl;
                loss_output_ << iter_ << '\t' << result << '\n';

                backward(loss);

                // Adjust gradients
                // ie. increment all parameters by 0.01 * p->grad()

                func_.adjust(learning_rate);

                ++iter_;
            }

            return result;
        }

    private:
        F func_;
        std::ofstream loss_output_;
        int iter_{0};
};

template<typename T, size_t N, typename Arg, typename F>
requires CanBackProp<F, T, Arg>
auto BackProp(const F& func, const std::string& loss_path)
{
    return BackPropImpl<T, N, Arg, F>(func, loss_path);
}

} // namespace ai
