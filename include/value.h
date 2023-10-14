#pragma once

#include <cmath>
#include <iostream>
#include <functional>
#include <memory>
#include <set>

namespace ai {

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
        static ptr add_label(ptr unlabelled, const std::string& label)
        {
            unlabelled->label_ = label;
            return unlabelled;
        }

        template <typename... Args>
        static ptr make(Args&&... args) {
            auto v = ptr(new RawValue<T>(std::forward<Args>(args)...));
            universe_.insert(v.get());
            return v;
        }

        const T& data() const {
            return data_;
        }

        double grad() const {
            return grad_;
        }

        void adjust(const T& learning_rate) {
            data_ += -learning_rate * grad_;
        }

        const std::string& label() const {
            return label_;
        }

        const std::set<ptr>& children() const {
            return prev_;
        }

        const std::string& op() const {
            return op_;
        }

        void zerograd() {
            grad_ = 0.0;
            for (auto c : prev_) {
                c->zerograd();
            }
        }

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

        // operator+
        friend ptr operator+(const ptr& a, const ptr& b) {
            std::set<ptr> children = {a, b};

            auto out = make(a->data() + b->data(), children, "+");

            out->backward_ = [=]() {
                a->grad_ += out->grad_;
                b->grad_ += out->grad_;
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator+(const ptr& a, N n) { return a + make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator+(N n, const ptr& a) { return make(n) + a; }

        // operator +=
        friend ptr& operator+=(ptr& a, const ptr& b) {
            /* For producing example-usage-cycle graph: cannot backprop
            a->data_ += b->data_;
            a->prev_.insert(b);
            return a;
            */

            // Update a = old + b

            auto old = make(a->data_, a->prev_, a->op_);
            old->grad_ = a->grad_;
            old->label_ = std::move(a->label_);
            old->backward_ = std::move(a->backward_);

            // Replace any instances of a with old
            for (auto p : universe_) {
                if (p->prev_.contains(a)) {
                    p->prev_.erase(a);
                    p->prev_.insert(old);
                }
            }

            a->grad_ = 0;
            a->data_ += b->data_;
            a->label_ = "";
            a->prev_ = {old, b};

            a->backward_ = [=]() {
                old->grad_ += a->grad_;
                b->grad_ += a->grad_;
            };

            return a;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr& operator+=(ptr& a, N n) { return a += make(n); }

        // unary operator-
        friend ptr operator-(const ptr& a) {
            std::set<ptr> children = {a};
            auto out = make(-a->data(), children, "neg");

            out->backward_ = [=]() {
                a->grad_ -= out->grad_;
            };

            return out;
        }

        // operator-
        friend ptr operator-(const ptr& a, const ptr& b) {
            std::set<ptr> children = {a, b};
            auto out = make(a->data() - b->data(), children, "-");

            out->backward_ = [=]() {
                a->grad_ += out->grad_;
                b->grad_ += -out->grad_;
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator-(const ptr& a, N n) { return a - make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator-(N n, const ptr& a) { return make(n) - a; }

        // operator*
        friend ptr operator*(const ptr& a, const ptr& b) {
            std::set<ptr> children = {a, b};
            auto out = make(a->data() * b->data(), children, "*");

            out->backward_ = [=]() {
                a->grad_ += b->data() * out->grad();
                b->grad_ += a->data() * out->grad();
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator*(const ptr& a, N n) { return a * make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator*(N n, const ptr& a) { return make(n) * a; }

        // operator/
        friend ptr operator/(const ptr& a, const ptr& b) {
            std::set<ptr> children = {a, b};
            auto out = make(a->data() / b->data(), children, "/");

            out->backward_ = [=]() {
                a->grad_ += (1.0/b->data()) * out->grad();
                b->grad_ += -1.0 * a->data() * pow(b->data(), -2.0) * out->grad();
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator/(const ptr& a, N n) { return a / make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator/(N n, const ptr& a) { return make(n) / a; }

        // recip
        friend ptr recip(const ptr& a) {
            std::set<ptr> children = {a};
            double t = pow(a->data(), -1.0);
            auto out = make(t, children, "recip");

            out->backward_ = [=]() {
                a->grad_ += (-1.0 * pow(a->data(), -2.0)) * out->grad();
            };

            return out;
        }

        // exp
        friend ptr exp(const ptr& a) {
            std::set<ptr> children = {a};
            double t = exp(a->data());
            auto out = make(t, children, "exp");

            out->backward_ = [=]() {
                a->grad_ += out->data_ * out->grad_;
            };

            return out;
        }

        // pow
        friend ptr pow(const ptr& a, const ptr& b) {
            std::set<ptr> children = {a, b};
            double t = pow(a->data(), b->data());
            auto out = make(t, children, "pow");

            out->backward_ = [=]() {
                a->grad_ += (b->data() * pow(a->data(), (b->data()-1))) * out->grad();
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr pow(const ptr& a, N n) { return pow(a, make(n)); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr pow(N n, const ptr& a) { return pow(make(n), a); }

        // tanh
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

        // relu
        friend ptr relu(const ptr& a) {
            std::set<ptr> children = {a};
            T x = a->data();
            T v = (x < 0) ? 0 : x;
            auto out = make(v, children, "relu");

            out->backward_ = [=]() {
                T t = out->data_;
                a->grad_ += (t > 0) ? out->grad_ : 0;
            };

            return out;
        }

    private:
        static std::set<RawValue<T>*> universe_;
        T data_;
        double grad_{0.0};
        std::string label_{};
        std::set<ptr> prev_{};
        std::string op_{""};

        std::function<void()> backward_{};
};

template <typename T>
inline std::set<RawValue<T>*> RawValue<T>::universe_ = {};

template <typename T>
using Value = typename RawValue<T>::ptr;

template <typename T, typename... Args>
static Value<T> make_value(const T& data, Args&&... args) {
    return RawValue<T>::make(data, std::forward<Args>(args)...);
}

template <typename T>
static Value<T> expr(std::shared_ptr<RawValue<T>> unlabelled, const std::string& label) {
    return RawValue<T>::add_label(unlabelled, label);
}

template <typename T, size_t N>
constexpr std::array<Value<T>, N> value_array(const std::array<T, N>& init) {
    std::array<Value<T>, N> result{};
    for (size_t i = 0; i < N; ++i) {
        result[i] = RawValue<T>::make(init[i]);
    }
    return result;
}

template <typename T, size_t M, size_t N>
constexpr std::array<Value<T>, N> value_arrays(const std::array<std::array<T, M>, N>& init) {
    std::array<std::array<Value<T>, M>, N> result{};
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; i < M; ++i) {
            result[j][i] = RawValue<T>::make(init[j][i]);
        }
    }
    return result;
}

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const RawValue<T>& value) {
    return os << "Value("
        << "label=" << value.label() << ", "
        << "data=" << value.data() << ", "
        << "grad=" << value.grad() << ", "
        << "op=" << value.op()
        << ")";
}

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<RawValue<T>>& value) {
    return os << value.get() << "=&" << *value;
}

}
