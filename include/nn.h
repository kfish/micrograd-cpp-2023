#pragma once

#include <cmath>

#include "array.h"
#include "funcy.h"
#include "tuple.h"
#include "quickmath.h"
#include "randomdata.h"
#include "static_index.h"
#include "value.h"

namespace ai {

template <typename T, size_t Nin>
class Neuron {
    public:
        Neuron()
            : weights_(randomArray<T, Nin>()), bias_(randomValue<T>())
        {
#ifdef DEBUG
            std::cerr << "Cnstrct " << this << "=&Neuron<" << Nin << "> "
                << *this << std::endl;
#endif
        }

        //Value<T> operator()(const std::array<Value<T>, Nin>& x, size_t nix=0) const {
        Value<T> operator()(const std::array<Value<T>, Nin>& x) const {
            //std::cerr << "Neuron(): this=" << this << std::endl;
            // y = w*x + b
            //std::cerr << "Neuron(): input x=" << PrettyArray(x) << std::endl;
            auto zero = leaf<double>(0.0);
#if 1
            Value<T> y = mac(weights_, x, zero) + bias_;
#else
            std::cerr << "\tbias=" << bias_ << std::endl;
            //Value<T> y = bias_;
            //auto it = x.begin();
            size_t i = 0;
#if 1
            for (auto & w : weights_) {
                //auto weight = expr(w, "w" + std::to_string(i));
                //std::cerr << "       *: " << w << " * " << *it << std::endl;
                //std::cerr << "       *: " << w << std::endl;
                std::cerr << "       *: " << *it << std::endl;
                //y = y + expr(*it * w, "y" + std::to_string(i));
                //y = y + *it;
                ++i;
            }
#endif
            return zero;
#endif
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
        Layer()
        {
            //std::cerr << "Cnstrct " << this << "=&" << *this << std::endl;
        }

        std::array<Value<T>, Nout> operator()(const std::array<Value<T>, Nin>& x) {
            // map_array_with_index...
            return map_array<Neuron<T, Nin>, std::array<Value<T>, Nin>, Value<T>, Nout>(neurons_, x);
        }

        const std::array<Neuron<T, Nin>, Nout> neurons() const {
            return neurons_;
        }

        void adjust_weights(const T& learning_rate) {
            for (auto & n : neurons_) {
                n.adjust_weights(learning_rate);
            }
        }

        void adjust_bias(const T& learning_rate) {
            for (auto & n : neurons_) {
                n.adjust_bias(learning_rate);
            }
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
void layers_adjust_weights(std::tuple<Ls...>& layers, const T& learning_rate) {
    std::apply([&learning_rate](auto&... layer) {
        // Use fold expression to call adjust_weights on each layer
        (..., layer.adjust_weights(learning_rate));
    }, layers);
}

template<typename T, typename... Ls>
void layers_adjust_bias(std::tuple<Ls...>& layers, const T& learning_rate) {
    std::apply([&learning_rate](auto&... layer) {
        // Use fold expression to call adjust_bias on each layer
        (..., layer.adjust_bias(learning_rate));
    }, layers);
}

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

    MLP() {
        //init<0, Nin, Nouts...>();
        //std::cerr << "Cnstrct " << this << "=&" << *this << std::endl;
    }

    const Layers<T, Nin, Nouts...>& layers() const
    {
        return layers_;
    };

    std::array<Value<T>, Nout> operator()(const std::array<Value<T>, Nin>& input) {
        //std::cerr << "mlp val: this=" << this << std::endl;
        return forward<0, Nin, Nouts...>(input);
    }

    std::array<Value<T>, Nout> operator()(const std::array<T, Nin>& input) {
        //std::cerr << "mlp raw: this=" << this << std::endl;
        return this->operator()(value_array(input));
    }

    void adjust_weights(const T& learning_rate) {
    }

    void adjust_bias(const T& learning_rate) {
    }

    void adjust(const T& learning_rate) {
        layers_adjust(layers_, learning_rate);
    }

private:
#if 0
    template <size_t I, size_t NinCurr, size_t NoutCurr, size_t... NoutsRest>
    void init() {
        static_assert(I < sizeof...(Nouts), "Invalid index.");
        std::cerr << "MLP: init layer " << I << std::endl;
        std::get<I>(layers_) = Layer<T, NinCurr, NoutCurr>{};
        if constexpr (sizeof...(NoutsRest) > 0) {
            init<I + 1, NoutCurr, NoutsRest...>();
        }
    }
#endif

    template <size_t I, size_t NinCurr, size_t NoutCurr, size_t... NoutsRest>
    auto forward(const std::array<Value<T>, NinCurr>& input) -> decltype(auto) {
        //std::cerr << "MLP forward: this=" << this << std::endl;
        auto & p = std::get<I>(layers_);
        //std::cerr << "forward<" << I << ">: " << &p << std::endl;
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
