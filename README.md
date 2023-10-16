# micrograd-cpp-2023

A C++ implementation of
[karpathy/micrograd](https://github.com/karpathy/micrograd).
Each step of the first episode of *Neural Nets: Zero to Hero*:
[The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0)
is included.

## What is micrograd-cpp and why is it interesting?

micrograd-cpp introduces some nuances of backpropagation (reverse-mode autodiff) and its
implementation. At its core is an expression graph which can be evaluated forwards, where
an expression like `a+b` is implemented using `operator+`, and differentiated in reverse
using a `std::function` attached to each graph node to calculate its gradient.

This implementation allows computations using `Value` objects to be written as
normal-looking C++ code.


### Example usage

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
so let's not limit ourselves to `float` or `double`. We template using `Value<T>`.

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
However, the `Value` objects in `micrograd` don't implement `__iadd__`, so Python falls back to using `__add__` followed by assignment. That means `a += b` is roughly equivalent to `a = a + b`. Each time the + operator is invoked, a new Value object is created and the computational graph gets extended, so it is not modifying the existing objects in-place from a computational graph perspective. Rather, it's creating new nodes in the graph, extending it with more operations to backtrack during the backward() pass.

In C++, `operator+=` requires an explicit implementation which modifies its value in-place.
We create a copy of the old value and re-write all earlier references in the expression graph
to point to the copy.

![c += c + 1](examples/c-plus-equals-rewrite.svg)

Note that this aspect of the implementation is peculiar to the operational semantics of C++
and in-place assignment operators. It is straightforward to implement a neural network
without calling these operators, so the overhead of node copying and graph rewriting could
easily be removed. We include it here only for the translation of micrograd to C++.

## Buliding out the Value object

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

>  For each value, we're going to calculate the derivative with respect to L.

We add a variable `grad_` inside the Value class that maintains the gradient with respect to L.

How each operation affects the output
`backward_` function, which we implement as a lambda function.
This function copies the `Value` `shared_ptr`s in order to increment their reference counts.

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
Recursively apply the local derivatives using the chain rule backwards through the expression graph

        `friend void backward(const ptr& node)`

## Backpropagation through a neuron

Introduce activation function, using tanh.

and also pow, exp, division

```c++
        friend ptr tanh(const ptr& a) {
            std::set<ptr> children = {a};
            double x = a->data();
            double e2x = exp(2.0*x);
            double t = (e2x-1)/(e2x+1);
            auto out = make(t, children, "tanh");

            out->backward_ = [=]() {
                T t = out->data_;
                a->grad_ += (1.0 - t*t) * out->grad_;
            };

            return out;
        }
```

Operator specializations where one operand is an arithmetic value, so that instead of
writing `a * make_value(7.0)` you can write `a * 7.0`:

```c++
        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator*(const ptr& a, N n) { return a * make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator*(N n, const ptr& a) { return make(n) * a; }
```

We begin the implementation of a neurons, in [include/nn.h](include/nn.h).

```c++
template <typename T, size_t Nin>
class Neuron {
    public:
        Neuron()
            : weights_(randomArray<T, Nin>()), bias_(randomValue<T>())
        {
        }

        Value<T> operator()(const std::array<Value<T>, Nin>& x) const {
            auto zero = make_value<T>(0.0);
            Value<T> y = mac(weights_, x, zero) + bias_;
            return expr(tanh(y), "n");
        }

        ...
};
```

![Neuron graph](examples/neuron.svg)

## Multiply-Accumulate

The implementation is in [include/mac.h](include/mac.h).

Use `std::execution` to allow the compiler to choose an optimized execution method,
allowing parallel and vectorized execution:

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

## RandomArray

In order to instantiate neurons statically
create random values in deterministic order, for reproducible debugging.

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

## References

### Automatic differentiation in C++
* https://www.youtube.com/watch?v=1QQj1mAV-eY
* https://compiler-research.org/assets/presentations/CladInROOT_15_02_2020.pdf
* https://arxiv.org/abs/2102.03681

### Automatic differentiation
* https://hackage.haskell.org/package/ad


