#pragma once

#include <cmath>

#include "array.h"
#include "tuple.h"
#include "mac.h"
#include "randomdata.h"
#include "value.h"

namespace ai {

template <typename T, size_t Nin>
class Neuron {
    public:
        Neuron()
            : weights_(randomArray<T, Nin>()), bias_(randomValue<T>())
        {
        }

        Value<T> operator()(const std::array<Value<T>, Nin>& x) const {
            Value<T> y = mac(weights_, x, bias_);
            return expr(tanh(y), "n");
        }

        const std::array<Value<T>, Nin>& weights() const {
            return weights_;
        }

        Value<T> bias() const {
            return bias_;
        }

        void adjust_weights(const T& learning_rate) {
            for (const auto& w : weights_) {
                w->adjust(learning_rate);
            }
        }

        void adjust_bias(const T& learning_rate) {
            bias_->adjust(learning_rate);
        }

        void adjust(const T& learning_rate) {
            adjust_weights(learning_rate);
            adjust_bias(learning_rate);
        }

    private:
        std::array<Value<T>, Nin> weights_{};
        Value<T> bias_{};
};

template <typename T, size_t Nin>
static inline std::ostream& operator<<(std::ostream& os, const Neuron<T, Nin>& n) {
    return os << "Neuron(" << PrettyArray(n.weights()) << "  bias=" << n.bias() << ")";
}

template <typename T, size_t Nin, size_t Nout>
class Layer {
    public:
        std::array<Value<T>, Nout> operator()(const std::array<Value<T>, Nin>& x) const {
            std::array<Value<T>, Nout> output{};
            std::transform(std::execution::par_unseq, neurons_.begin(), neurons_.end(),
                    output.begin(), [&](const auto& n) { return n(x); });
            return output;
        }

        const std::array<Neuron<T, Nin>, Nout> neurons() const {
            return neurons_;
        }

        void adjust(const T& learning_rate) {
            for (auto & n : neurons_) {
                n.adjust(learning_rate);
            }
        }

    private:
        std::array<Neuron<T, Nin>, Nout> neurons_{};
};

template <typename T, size_t Nin, size_t Nout>
static inline std::ostream& operator<<(std::ostream& os, const Layer<T, Nin, Nout>& l) {
    return os << "Layer(" << PrettyArray(l.neurons()) << ")";
}

template <typename T, size_t Nin, size_t... Nouts>
struct BuildLayers;

template <typename T, size_t Nin, size_t First, size_t... Rest>
struct BuildLayers<T, Nin, First, Rest...> {
    using type = decltype(std::tuple_cat(
        std::tuple<Layer<T, Nin, First>>{},
        typename BuildLayers<T, First, Rest...>::type{}
    ));
    static constexpr size_t nout = BuildLayers<T, First, Rest...>::nout;
};

template <typename T, size_t Nin, size_t Last>
struct BuildLayers<T, Nin, Last> {
    using type = std::tuple<Layer<T, Nin, Last>>;
    static constexpr size_t nout = Last;
};

template <typename T, size_t Nin, size_t... Nouts>
using Layers = typename BuildLayers<T, Nin, Nouts...>::type;

template <typename T, size_t Nin, size_t... Nouts>
static constexpr size_t LayersNout = BuildLayers<T, Nin, Nouts...>::nout;

template<typename T, typename... Ls>
void layers_adjust(std::tuple<Ls...>& layers, const T& learning_rate) {
    std::apply([&learning_rate](auto&... layer) {
        // Use fold expression to call adjust on each layer
        (..., layer.adjust(learning_rate));
    }, layers);
}

template <typename T, size_t Nin, size_t... Nouts>
class MLP {
public:
    static constexpr size_t Nout = LayersNout<T, Nin, Nouts...>;

    const Layers<T, Nin, Nouts...>& layers() const
    {
        return layers_;
    };

    std::array<Value<T>, Nout> operator()(const std::array<Value<T>, Nin>& input) const {
        return forward<0, Nin, Nouts...>(input);
    }

    std::array<Value<T>, Nout> operator()(const std::array<T, Nin>& input) const {
        return this->operator()(value_array(input));
    }

    void adjust(const T& learning_rate) {
        layers_adjust(layers_, learning_rate);
    }

private:
    template <size_t I, size_t NinCurr, size_t NoutCurr, size_t... NoutsRest>
    auto forward(const std::array<Value<T>, NinCurr>& input) const -> decltype(auto) {
        auto & p = std::get<I>(layers_);
        auto output = std::get<I>(layers_)(input);
        if constexpr (sizeof...(NoutsRest) > 0) {
            return forward<I + 1, NoutCurr, NoutsRest...>(output);
        } else {
            return output;
        }
    }

private:
    Layers<T, Nin, Nouts...> layers_;

};

template <typename T, size_t Nin, size_t... Nouts>
static inline std::ostream& operator<<(std::ostream& os, const MLP<T, Nin, Nouts...>& mlp) {
    os << "MLP";
    print_args<Nin, Nouts...>(os);
    os << ": ";
    print_tuple(os, mlp.layers());
    os << ")";
    return os;
}

// MLP that only considers the first output
template <typename T, size_t Nin, size_t... Nouts>
class MLP1 : public MLP<T, Nin, Nouts...>
{
    public:
        MLP1()
            : MLP<T, Nin, Nouts...>()
        {}

        Value<T> operator()(const std::array<Value<T>, Nin>& input) {
            return MLP<T, Nin, Nouts...>::operator()(input)[0];
        }

        Value<T> operator()(const std::array<T, Nin>& input) {
            return MLP<T, Nin, Nouts...>::operator()(input)[0];
        }
};

}
