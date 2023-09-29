#pragma once

#include <cmath>
#include <tuple>

#include "array.h"
#include "funcy.h"
#include "quickmath.h"
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
            // y = w*x + b
            auto zero = leaf<double>(0.0);
            Value<T> y = mac(weights_, x, zero) + bias_;
            return tanh(y);
        }

        const std::array<Value<T>, Nin>& weights() const {
            return weights_;
        }

        Value<T> bias() const {
            return bias_;
        }

    private:
        std::array<Value<T>, Nin> weights_{};
        Value<T> bias_{};
};

template <typename T, size_t Nin>
static inline std::ostream& operator<<(std::ostream& os, const Neuron<T, Nin>& n) {
    return os << "NN(" << PrettyArray(n.weights()) << " bias=" << n.bias() << ")";
}

template <typename T, size_t Nin, size_t Nout>
class Layer {
    public:
        std::array<Value<T>, Nout> operator()(const std::array<Value<T>, Nin>& x) {
            return map_array<Neuron<T, Nin>, std::array<Value<T>, Nin>, Value<T>, Nout>(neurons_, x);
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

template <typename T, size_t Nin, size_t... Nouts>
struct BuildLayers;

template <typename T, size_t Nin, size_t First, size_t... Rest>
struct BuildLayers<T, Nin, First, Rest...> {
    using type = decltype(std::tuple_cat(
        std::tuple<Layer<T, Nin, First>>{},
        typename BuildLayers<T, First, Rest...>::type{}
    ));
};

template <typename T, size_t Nin, size_t Last>
struct BuildLayers<T, Nin, Last> {
    using type = std::tuple<Layer<T, Nin, Last>>;
};

template <typename T, size_t Nin, size_t... Nouts>
using Layers = typename BuildLayers<T, Nin, Nouts...>::type;

template <typename T, size_t Nin, size_t... Nouts>
class MLP {
public:
    MLP() {
        init<0, Nin, Nouts...>();
    }

    const Layers<T, Nin, Nouts...>& layers() const
    {
        return layers_;
    };

    auto operator()(const std::array<Value<T>, Nin>& input) {
        return forward<0, Nin, Nouts...>(input);
    }

    auto operator()(const std::array<T, Nin>& input) {
        return this->operator()(value_array(input));
    }

private:
    template <size_t I, size_t NinCurr, size_t NoutCurr, size_t... NoutsRest>
    void init() {
        static_assert(I < sizeof...(Nouts), "Invalid index.");
        std::get<I>(layers_) = Layer<T, NinCurr, NoutCurr>{};
        if constexpr (sizeof...(NoutsRest) > 0) {
            init<I + 1, NoutCurr, NoutsRest...>();
        }
    }

    template <size_t I, size_t NinCurr, size_t NoutCurr, size_t... NoutsRest>
    auto forward(const std::array<Value<T>, NinCurr>& input) -> decltype(auto) {
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

// Recursive template function to print each layer in a tuple
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), std::ostream&>::type
print_tuple(std::ostream& os, const std::tuple<Tp...>& t)
{
    return os;
}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), std::ostream&>::type
print_tuple(std::ostream& os, const std::tuple<Tp...>& t)
{
    os << std::get<I>(t);
    if (I + 1 != sizeof...(Tp)) os << ", ";
    return print_tuple<I + 1, Tp...>(os, t);
}

// Overload for MLP
template <typename T, size_t Nin, size_t... Nouts>
static inline std::ostream& operator<<(std::ostream& os, const MLP<T, Nin, Nouts...>& mlp) {
    os << "MLP(";
    print_tuple(os, mlp.layers());
    os << ")";
    return os;
}

}
