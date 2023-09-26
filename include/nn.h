#pragma once

#include <cmath>

#include "value.h"
#include "funcy.h"
#include "quickmath.h"
#include "randomdata.h"

namespace ai {

template <typename T, size_t Nin>
class Neuron {
    public:
        Neuron()
            : weights_(randomArray<T, Nin>()), bias_(randomValue<T>())
        {
        }

        T operator()(const std::array<T, Nin>& x) const {
            // y = w*x + b
            T y = mac(weights_, x) + bias_;
            return tanh(y);
        }

        const std::array<T, Nin>& weights() const {
            return weights_;
        }

        T bias() const {
            return bias_;
        }

    private:
        std::array<T, Nin> weights_{};
        T bias_{};
};

template <typename T, size_t Nin>
static inline std::ostream& operator<<(std::ostream& os, const Neuron<T, Nin>& n) {
    return os << "NN(" << PrettyArray(n.weights()) << " bias=" << n.bias() << ")";
}

template <typename T, size_t Nin, size_t Nout>
class Layer {
    public:
        std::array<T, Nout> operator()(const std::array<T, Nin>& x) {
            return map_array<Neuron<T, Nin>, std::array<T, Nin>, T, Nout>(neurons_, x);
        }

        const std::array<Neuron<T, Nin>, Nout> neurons() const {
            return neurons_;
        }

    private:
        std::array<Neuron<T, Nin>, Nout> neurons_{};
};

template <typename T, size_t Nin, size_t Nout>
static inline std::ostream& operator<<(std::ostream& os, const Layer<T, Nin, Nout>& l) {
    return os << "Layer(" << PrettyArray(l.neurons()) << ")";
}

}
