# micrograd-cpp-2023

A C++ implementation of
[karpathy/micrograd](https://github.com/karpathy/micrograd).
Each step of the first episode of *Neural Nets: Zero to Hero*:
[The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0)
is included.

This roughly follows the flow of Karpathy's YouTube tutorial, with details specific to this C++ implementation:

 * [What is micrograd-cpp and why is it interesting?](#what-is-micrograd-cpp-and-why-is-it-interesting)
   - [Example usage](#example-usage)
   - [C++ implementation notes](#c-implementation-notes)
 * [Building out the Value object](#building-out-the-value-object)
 * [Visualizing the expression graph](#visualizing-the-expression-graph)
 * [Backpropagation](#backpropagation)
 * [Backpropagation through a neuron](#backpropagation-through-a-neuron)
   - [Activation function](#activation-function)
   - [Math operations](#math-operations)
   - [Multiply-Accumulate](#multiply-accumulate)
   - [randomValue, randomArray](#randomvalue-randomarray)
 * [Multi-Layer Perceptron](#multi-layer-perceptron)
   - [Layer](#layer)
   - [BuildLayers](#buildlayers)
   - [MLP](#mlp)
   - [MLP1](#mlp1)
 * [Loss function](#loss-function)
   - [MSELoss](#mseloss)
 * [Gradient descent](#gradient-descent)
   - [Adjusting parameters](#adjusting-parameters)
   - [CanBackProp](#canbackprop)
   - [BackProp](#backprop)
   - [Binary Classifier](#binary-classifier)

with [References](#references) at the end for further reading about automatic differentiation and C++ implementations.

## What is micrograd-cpp and why is it interesting?

micrograd-cpp introduces some nuances of backpropagation (reverse-mode autodiff) and its
implementation. At its core is an expression graph which can be evaluated forwards, where
an expression like `a+b` is implemented using `operator+`, and differentiated in reverse
using a `std::function` attached to each graph node to calculate its gradient.

This implementation allows computations using `Value` objects to be written as
normal-looking C++ code. It also includes generic classes for evaluation and learning for anything that can produce `Value<T>`.

Like micrograd, the point is still to be educational, with a focus on some implementation details
and flexibility for exploring learning algorithms.

### Building

For simplicity, there are no code dependencies. Data is manipulated using `std::array<>` and graphing and plotting is done
with external tools like `dot` and `gnuplot`.

Build with CMake, eg

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### Example usage

This is [examples/example-usage.cpp](examples/example-usage.cpp):

```c++
#include <iostream>

#include "value.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = make_value(-4.0);
    auto b = make_value(2.0);

    auto c = a + b;
    auto d = a * b + pow(b, 3);

    c += c + 1;
    c += 1 + c + (-a);
    d += d * 2 + relu(b + a);
    d += 3 * d + relu(b - a);
    auto e = c - d;
    auto f = pow(e, 2);
    auto g = f / 2.0;
    g += 10.0 / f;
    printf("%.4f\n", g->data()); // prints 24.7041, the outcome of this forward pass
    backward(g);
    printf("%.4f\n", a->grad()); // prints 138.8338, i.e. the numerical value of dg/da
    printf("%.4f\n", b->grad()); // prints 645.5773, i.e. the numerical value of dg/db
}
```

### C++ implementation notes

1. Data type

Neural nets generally don't require many bits of precision on individual node values,
so let's not limit ourselves to `float` or `double`
(or [FP8](https://github.com/opencomputeproject/FP8/blob/main/ofp8_references.pdf)).
We template using `Value<T>`.

2. Sharing

Nodes may appear as inputs to multiple other nodes in the expression graph,
especially for neural networks, so we use a `shared_ptr`:

```c++
using Value<T> = std::shared_ptr<RawValue<T>>;
```

3. Removal of cycles

The expression `c += c + 1` refers to itself, so it contains a cycle. This cycle needs to be
removed in order to implement backpropagation.

![c += c + 1](examples/c-plus-equals-cycle.svg)

In Python, `x += y` usually translates to `x.__iadd__(y)` which modifies `x` in-place.
However, the `Value` objects in `micrograd` don't implement `__iadd__`, so Python falls back to using `__add__`
followed by assignment. That means `a += b` is roughly equivalent to `a = a + b`. Each time the + operator
is invoked, a new Value object is created and the graph gets extended, so it is not modifying the existing
objects in-place.

In C++, `operator+=` requires an explicit implementation which modifies its value in-place.
We create a copy of the old value and re-write all earlier references in the expression graph
to point to the copy.

![c += c + 1](examples/c-plus-equals-rewrite.svg)

Note that this aspect of the implementation is peculiar to the operational semantics of C++
and in-place assignment operators. It is straightforward to implement a neural network
without calling these operators, so the overhead of node copying and graph rewriting could
easily be removed. We include it here only for the translation of micrograd to C++.

## Building out the Value object

> Neural nets are some pretty scary expressions. We need some data structures to maintain 
> these expressions.

In order to handle basic expressions like:

```c++
    auto a = make_value(2.0, "a");
    auto b = make_value(-3.0, "b");
    auto c = make_value(10.0, "c");

    auto d = (a*b) + c;
    std::cout << d << std::endl;
```

we start sketching out the underlying `RawValue` class, implementing operators for `+`
and `*`, and storing the inputs (children) of each for the evaluation graph.

```c++
template <typename T>
class RawValue {
    public:
        using ptr = std::shared_ptr<RawValue<T>>;

    private:
        RawValue(const T& data, const std::string& label="")
            : data_(data), label_(label)
        {}

        RawValue(const T& data, std::set<ptr>& children, const std::string& op="")
            : data_(data), prev_(children), op_(op)
        {}

    public:
        template <typename... Args>
        static ptr make(Args&&... args) {
            return ptr(new RawValue<T>(std::forward<Args>(args)...));
        }

        friend ptr operator+(const ptr& a, const ptr& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() + b->data(), children, "+");
        }

        friend ptr operator*(const ptr& a, const ptr& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() * b->data(), children, "*");
        }

    private:
        T data_;
        std::set<ptr> prev_{};
        std::string op_{""};
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const RawValue<T>& value) {
    return os << "Value("
        << "data=" << value.data() << ", "
        << "op=" << value.op()
        << ")";
}
```

In code we use `Value<T>`, which is an alias for `shared_ptr<RawValue<T>>`:

```c++
template <typename T>
using Value = typename RawValue<T>::ptr;

template <typename T, typename... Args>
static Value<T> make_value(const T& data, Args&&... args) {
    return RawValue<T>::make(data, std::forward<Args>(args)...);
}

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<RawValue<T>>& value) {
    return os << value.get() << "=&" << *value;
}
```

## Visualizing the expression graph

We provide a `Graph` class that can wrap any `Value<T>`. It has a custom `operator<<` that writes in `dot`
language. The implementation is in [include/graph.h](include/graph.h). We also introduce a `label` to the `Value<T>`
object for labelling graph nodes, and an `expr` factory function for creating labelled expressions.

We can pipe the output of a program to `dot -Tsvg` to produce an svg image, or to `xdot` to view it interactively:

```bash
$ build/examples/graph | dot -Tsvg -o graph.svg
$ build/examples/graph | xdot -
```

![Example graph](examples/graph.svg)

## Backpropagation

We add a member variable `grad_` that maintains the gradient with respect to the final output.

How each operation affects the output is written as a lambda function, `backward_`.
It copies the `Value` `shared_ptr`s of each node's children in order to increment their reference counts.

```c++
        friend ptr operator+(const ptr& a, const ptr& b) {
            auto out = make(a->data() + b->data(), children, "+");

            out->backward_ = [=]() {
                a->grad_ += out->grad_;
                b->grad_ += out->grad_;
            };

            return out;
        }

        friend ptr operator*(const ptr& a, const ptr& b) {
            std::set<ptr> children = {a, b};
            auto out = make(a->data() * b->data(), children, "*");

            out->backward_ = [=]() {
                a->grad_ += b->data() * out->grad();
                b->grad_ += a->data() * out->grad();
            };

            return out;
        }
```

We recursively apply the local derivatives using the chain rule backwards through the expression graph:

```c++
        friend void backward(const ptr& node) {
            std::vector<RawValue<T>*> topo;
            std::set<RawValue<T>*> visited;

            std::function<void(const ptr&)> build_topo = [&](const ptr& v) {
                if (!visited.contains(v.get())) {
                    visited.insert(v.get());
                    for (auto && c : v->children()) {
                        build_topo(c);
                    }
                    topo.push_back(v.get());
                }
            };

            build_topo(node);

            for (auto & v : topo) {
                v->grad_ = 0.0;
            }

            node->grad_ = 1.0;

            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                const RawValue<T>* v = *it;
                auto f = v->backward_;
                if (f) f();
            }
        }
```

## Backpropagation through a neuron

We begin the implementation of a neuron, in [include/nn.h](include/nn.h):

```c++
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

        ...
};
```

The resulting expression graph for a neuron with four inputs (code in [examples/neuron.cpp](examples/neuron.cpp)):

![Neuron graph](examples/neuron.svg)

### Activation function

In general an activation function modifies the output of a neuron, perhaps so that all neurons have similar ranges of output value or to smooth or filter large and negative values.
Whichever activation function we use, we need to implement a `backward_` function.
This implementation includes `relu` (which just replaces any negative values with zero) and `tanh`, which squashes the output into the range ±1.0. `tanh` is used in the video and has an obvious and continuous effect on the gradient:

```c++
        friend ptr tanh(const ptr& a) {
            std::set<ptr> children = {a};
            double x = a->data();
            double e2x = exp(2.0*x);
            double t = (e2x-1)/(e2x+1);
            auto out = make(t, children, "tanh");

            out->backward_ = [=]() {
                a->grad_ += (1.0 - t*t) * out->grad_;
            };

            return out;
        }
```

### Math operations

We must implement all required math operations on `Value<T>`, including pow, exp, and division,
so that we can accumulate gradients and run backpropagation.

For convenience we also provide operator specializations where one operand is an arithmetic value, so that instead of
writing `a * make_value(7.0)` you can write `a * 7.0` or `7.0 * a`:

```c++
        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator*(const ptr& a, N n) { return a * make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator*(N n, const ptr& a) { return make(n) * a; }
```

### Multiply-Accumulate

A neuron takes a number of input values, applies a weight to each, and sums the result. We can abstract this out as a common multiply-accumulate function.
It is usual to use a hardware-optimized, eg. GPU, implementation.
In order to use our explicit `Value` object, we provide a generic implementation is in [include/mac.h](include/mac.h).
This uses `std::execution` to allow the compiler to choose an optimized execution method, allowing parallel and vectorized execution:

```c++
template <typename T, std::size_t N>
T mac(const std::array<T, N>& a, const std::array<T, N>& b, T init = T{}) {
    return std::transform_reduce(
        std::execution::par_unseq, // Use parallel and vectorized execution
        a.begin(), a.end(), // Range of first vector
        b.begin(), // Range of second vector
        init, //static_cast<T>(0), // Initial value
        std::plus<>(), // Accumulate
        std::multiplies<>() // Multiply
    );
}
```

### randomValue, randomArray

We provide helper functions to create random values statically, in deterministic order. This helps with reproducibility for debugging.

The implementation is in [include/random.h](include/random.h).

```c++
// Static inline function to generate a random T
template <typename T>
static inline Value<T> randomValue() {
    static unsigned int seed = 42;
    static thread_local std::mt19937 gen(seed++);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    seed = gen(); // update seed for next time
    return make_value(dist(gen));
}

// Static inline function to generate a random std::array<T, N>
template <typename T, size_t N>
static inline std::array<Value<T>, N> randomArray() {
    std::array<Value<T>, N> arr;
    for (auto& element : arr) {
        element = randomValue<T>();
    }
    return arr;
}
```

## Multi-Layer Perceptron

We arrange neurons in a series of layers. Each layer is just an array of neurons.

A layer `Layer<T, Nin, Nout>` consists of `Nout` neurons, and is callable:
  * The same input (array of `Nin` values) is passed to each of the neurons
  * Each neuron produces a single output value
  * These output values are collected into an output array of `Nout` values.

### Layer

```c++
template <typename T, size_t Nin, size_t Nout>
class Layer {
    public:

        std::array<Value<T>, Nout> operator()(const std::array<Value<T>, Nin>& x) {
            return map_array<Neuron<T, Nin>, std::array<Value<T>, Nin>, Value<T>, Nout>(neurons_, x);

    private:
        std::array<Neuron<T, Nin>, Nout> neurons_{};
};
```

### BuildLayers

We introduce a helper type that allows us to specify a sequence of layers of different sizes.

```c++
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
```

We make an alias for the type of such a sequence, like `Layers<3, 4, 4, 1>`:

```c++
template <typename T, size_t Nin, size_t... Nouts>
using Layers = typename BuildLayers<T, Nin, Nouts...>::type;
```

and a helper to extract the final number of outputs, eg. `LayersNout<3, 4, 4, 1>` is 1:

```c++
template <typename T, size_t Nin, size_t... Nouts>
static constexpr size_t LayersNout = BuildLayers<T, Nin, Nouts...>::nout;
```

### MLP

Finaly we use `Layers<>` in a class `MLP<>`, which:
  * Forwards its input to the first layer
  * Passes the output of each layer to the next layer, in turn
  * Returns the output of the last layer

```c++
template <typename T, size_t Nin, size_t... Nouts>
class MLP {
public:
    static constexpr size_t Nout = LayersNout<T, Nin, Nouts...>;

    std::array<Value<T>, Nout> operator()(const std::array<Value<T>, Nin>& input) {
        return forward<0, Nin, Nouts...>(input);
    }

    std::array<Value<T>, Nout> operator()(const std::array<T, Nin>& input) {
        return this->operator()(value_array(input));
    }

private:
    template <size_t I, size_t NinCurr, size_t NoutCurr, size_t... NoutsRest>
    auto forward(const std::array<Value<T>, NinCurr>& input) -> decltype(auto) {
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
```

### MLP1

If we want a single-valued output from our neural network, we create a wrapper class `MLP1<>` that returns only the first element
of the output of the wrapped `MLP<>`:

```c++
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
```

With the following code from [examples/mlp1.cpp](examples/mlp1.cpp) we can make a 3 layer neural net with 1 output,
run backpropagation over it and show the resulting expression graph:

```c++
    MLP1<double, 3, 4, 4, 1> n;

    std::array<double, 3> input = {{ 2.0, 3.0, -1.0 }};
    auto output = n(input);

    backward(output);

    std::cout << Graph(output) << std::endl;
```

![MLP1 graph](examples/mlp1.svg)

## Loss function

Now that we can make a neural net, run it forwards to produce a value and backwards to calculate gradients, we can begin adjusting it to learn.

We introduce generic evaluation and learning classes for anything that can produce `Value<T>`.

### MSELoss

We evaluate a prediction against a known "ground truth". The difference between these is the *Error*, and we
take the square of the error to approximate distance.
We average these out when considering an array of predictions and their ground truths. This is the Mean Squared Error.

Implementation in [include/loss.h](include/loss.h).

```c++
template <typename T>
Value<T> mse_loss(const Value<T>& predicted, const Value<T>& ground_truth) {
    static_assert(std::is_arithmetic<T>::value, "Type must be arithmetic");
    return pow(predicted - ground_truth, 2);
}
```

```c++
template<typename T, size_t N>
Value<T> mse_loss(const std::array<Value<T>, N>& predictions, const std::array<T, N>& ground_truth) {
    Value<T> sum_squared_error = std::inner_product(predictions.begin(), predictions.end(), ground_truth.begin(), make_value<T>(0),
        std::plus<>(),
        [](Value<T> pred, T truth) { return pow(pred - truth, 2); }
    );
    return sum_squared_error / make_value<T>(N);
}
```

We provide a wrapper class to calculate the Mean Squared Error of any `std::function<Value<T>(const Arg&)>`:

```c++
template<typename T, size_t N, typename Arg>
class MSELoss {
    public:
        MSELoss(const std::function<Value<T>(const Arg&)>& func)
            : func_(func)
        {
        }

        Value<T> operator()(std::array<Arg, N>& input, const std::array<T, N>& ground_truth, bool verbose=false) {
            if (verbose) std::cerr << "Predictions: ";
            for (size_t i = 0; i < N; ++i) {
                predictions_[i] = func_(input[i]);
                if (verbose) std::cerr << predictions_[i]->data() << " ";
            }
            if (verbose) std::cerr << '\n';
            return mse_loss(predictions_, ground_truth);
        }

    private:
        const std::function<Value<T>(const Arg&)> func_;
        std::array<Value<T>, N> predictions_;
};
```

## Gradient descent

The gradients calculated by running `backward(loss)` annotate how each parameter contributes to the error loss.
By adjusting each parameter down (against the gradient) we aim to minimize the error.

### Adjusting parameters

We introduce an `adjust()` function in `Value<T>` to modify `data_` according to the calculated gradient `grad_`.
This takes a parameter `learning_rate`, usually in the range `[0.0 .. 1.0]` to scale of the adjustment:

```c++
        void adjust(const T& learning_rate) {
            data_ += -learning_rate * grad_;
        }
```

We then provide `adjust()` functions to adjust all the parameters of an neural net: the weights and bias of a Neuron:

```c++
template <typename T, size_t Nin>
class Neuron {
    ...
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
    ...
};
```

then adjust all the Neurons in a Layer:

```c++
template <typename T, size_t Nin, size_t Nout>
class Layer {
    ...
        void adjust(const T& learning_rate) {
            for (auto & n : neurons_) {
                n.adjust(learning_rate);
            }
        }
    ...
};
```

and all the Layers in an MLP:

```c++
template<typename T, typename... Ls>
void layers_adjust(std::tuple<Ls...>& layers, const T& learning_rate) {
    std::apply([&learning_rate](auto&... layer) {
        // Use fold expression to call adjust on each layer
        (..., layer.adjust(learning_rate));
    }, layers);
}
```

```c++
template <typename T, size_t Nin, size_t... Nouts>
class MLP {
    ...
        void adjust(const T& learning_rate) {
            layers_adjust(layers_, learning_rate);
        }
    ...
};
```

### CanBackProp

We introduce a concept `CanBackProp` to describe any function that can be evaluated and adjusted:

```c++
template <typename F, typename T, typename Arg>
concept CanBackProp = requires(F f, Arg arg, T learning_rate) {
    { f(arg) } -> std::convertible_to<Value<T>>;
    { f.adjust(learning_rate) } -> std::convertible_to<void>;
};
```

For example, `CanBackProp` is true for `MLP1`.

### BackProp

We create a wrapper class for any function that matches the `CanBackProp` concept. This class is callable
with input and ground truth arguments, which are used to iteratively:
  * make predictions
  * evaluate error loss against ground truth
  * adjust parameters to minimize loss

The loss at each step is recorded in an output file `loss_path`.

```c++
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
                T learning_rate, int iterations, bool verbose=false)
        {
            auto loss_f = loss_function();
            T result;

            for (int i=0; i < iterations; ++i) {
                Value<T> loss = loss_f(input, ground_truth, verbose);

                result = loss->data();
                loss_output_ << iter_ << '\t' << result << '\n';

                if (verbose) {
                    std::cerr << "Loss (" << iter_ << "):\t" << result << std::endl;
                }

                backward(loss);

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
```

The helper `BackProp` template allows us to instantiate without specifying the type of `F`, as the compiler can infer it from the constructor argument:

```c++
template<typename T, size_t N, typename Arg, typename F>
requires CanBackProp<F, T, Arg>
auto BackProp(const F& func, const std::string& loss_path)
{
    return BackPropImpl<T, N, Arg, F>(func, loss_path);
}
```

### Binary Classifier

Full example: [binary-classifier.cpp](examples/binary-classifier.cpp):

```c++
#include <iostream>

#include "backprop.h"
#include "nn.h"

using namespace ai;

int main(int argc, char *argv[])
{
    // Define a neural net
    MLP1<double, 3, 4, 4, 1> n;

    std::cerr << n << std::endl;

    std::array<std::array<double,3>, 4> input = {{
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    }};

    std::array<double, 4> y = {1.0, -1.0, -1.0, 1.0};
    std::cerr << "y (gt):\t" << PrettyArray(y) << std::endl;

    // Run backprop
    double learning_rate = 0.9;

    auto backprop = BackProp<double, 4, std::array<double, 3>>(n, "loss.tsv");
    double loss = backprop(input, y, learning_rate, 20, true);
}
```

This quickly converges close to the ground truth `y = {1.0, -1.0, -1.0, 1.0}`:

```
Predictions: 0.773488 0.796802 0.870344 0.736159
Loss (0):	1.7119
Predictions: -0.316783 -0.714319 -0.588441 -0.396673
Loss (1):	0.983902
Predictions: 0.997675 0.996129 0.996903 0.99767
Loss (2):	1.99304
Predictions: 0.997454 0.995671 0.996576 0.997448
Loss (3):	1.99226
Predictions: 0.997183 0.995088 0.996169 0.997177
Loss (4):	1.99127
Predictions: 0.996845 0.99432 0.995649 0.996838
Loss (5):	1.98999
Predictions: 0.996409 0.993259 0.99496 0.996403
Loss (6):	1.98824
Predictions: 0.995827 0.991692 0.994 0.995819
Loss (7):	1.98573
Predictions: 0.995001 0.989122 0.992564 0.994993
Loss (8):	1.98174
Predictions: 0.993729 0.984072 0.990152 0.993718
Loss (9):	1.97433
Predictions: 0.991462 0.969564 0.985135 0.991443
Loss (10):	1.95502
Predictions: 0.985916 0.862539 0.967274 0.985854
Loss (11):	1.8349
Predictions: 0.947625 -0.182752 0.497795 0.945489
Loss (12):	0.72925
Predictions: -0.0306315 -0.996451 -0.996155 -0.512822
Loss (13):	0.837715
Predictions: 0.999406 -0.977287 -0.542644 0.999393
Loss (14):	0.0524229
Predictions: 0.999027 -0.997707 -0.997583 0.998998
Loss (15):	3.26197e-06
Predictions: 0.999027 -0.997707 -0.997584 0.998998
Loss (16):	3.26134e-06
Predictions: 0.999027 -0.997708 -0.997584 0.998998
Loss (17):	3.26072e-06
Predictions: 0.999027 -0.997708 -0.997584 0.998998
Loss (18):	3.26009e-06
Predictions: 0.999027 -0.997708 -0.997585 0.998998
Loss (19):	3.25946e-06
```

## Conclusion

We ported all the features of micrograd introduced in Karpathy's YouTube tutorial to C++, giving a different perspective on implementation details.
We also considered some more generic aspects of model evaluation and iterative learning to develop re-usable C++ classes.

## References

### Automatic differentiation in C++
* [Differentiable Programming in C++ - Vassil Vassilev & William Moses - CppCon 2021](https://www.youtube.com/watch?v=1QQj1mAV-eY) [YouTube]
* [Automatic Differentiation in C++](https://compiler-research.org/assets/presentations/CladInROOT_15_02_2020.pdf)[PDF]
* [FastAD: Expression Template-Based C++ Library for Fast and Memory-Efficient Automatic Differentiation](https://arxiv.org/abs/2102.03681) [PDF]

### Automatic differentiation
* [ad](https://hackage.haskell.org/package/ad)


