#pragma once

#include <cmath>
#include <iostream>
#include <functional>
#include <memory>
#include <set>

namespace ai {

template <typename T>
class RawValue;

template <typename T>
using Value = RawValue<T>::ptr;

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
            //std::cerr << "Labelled " << unlabelled << std::endl;
            return unlabelled;
        }

        template <typename... Args>
        static ptr make(Args&&... args) {
            return ptr(new RawValue<T>(std::forward<Args>(args)...));
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

        const std::set<ptr> children() const {
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

        friend void backward(const Value<T>& node) {
            std::vector<RawValue<T>*> topo;
            std::set<RawValue<T>*> visited;

            std::function<void(const Value<T>&)> build_topo = [&](const Value<T>& v) {
                if (!visited.contains(v.get())) {
                    visited.insert(v.get());
                    for (auto && c : v->children()) {
                        build_topo(c);
                    }
                    topo.push_back(v.get());
                }
            };

            build_topo(node);

            //std::cerr << "Built topo: size=" << topo.size() << std::endl;

            node->zerograd();
            node->grad_ = 1.0;

            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                const RawValue<T>* v = *it;
                //std::cerr << "Backprop: " << v << "=&" << *v << std::endl;
                auto f = v->backward_;
                if (f) f();
            }
        }

        // operator+
        friend Value<T> operator+(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};

            auto out = make(a->data() + b->data(), children, "+");

#if 0
            std::cerr << "Made + node of:\n\t" << a << "\n  +\t" << b
                << "\n  out=\t" << out << std::endl;
#endif

            out->backward_ = [=]() {
                a->grad_ += out->grad_;
                b->grad_ += out->grad_;

#if 0
                std::cerr << "  +.grad: out=" << out
                    << "\n  upd a=\t" << a
                    << "\n  upd b=\t" << b
                    << std::endl;
#endif
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> operator+(const Value<T>& a, N n) { return a + make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> operator+(N n, const Value<T>& a) { return make(n) + a; }

        // unary operator-
        friend Value<T> operator-(const Value<T>& a) {
            std::set<ptr> children = {a};
            auto out = make(-a->data(), children, "neg");

            out->backward_ = [=]() {
                a->grad_ -= out->grad_;
            };

            return out;
        }

        // operator-
        friend Value<T> operator-(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            auto out = make(a->data() - b->data(), children, "-");

            out->backward_ = [=]() {
                a->grad_ += out->grad_;
                b->grad_ += -out->grad_;
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> operator-(const Value<T>& a, N n) { return a - make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> operator-(N n, const Value<T>& a) { return make(n) - a; }

        // operator*
        friend Value<T> operator*(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            auto out = make(a->data() * b->data(), children, "*");

            out->backward_ = [=]() {
                a->grad_ += b->data() * out->grad();
                b->grad_ += a->data() * out->grad();
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> operator*(const Value<T>& a, N n) { return a * make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> operator*(N n, const Value<T>& a) { return make(n) * a; }

        // operator/
#if 1
        friend Value<T> operator/(const Value<T>& a, const Value<T>& b) {
            //return a * recip(b);
            return a * pow(b, -1.0);
        }
#else
        friend Value<T> operator/(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            auto out = make(a->data() / b->data(), children, "/");

            out->backward_ = [=]() {
                a->grad_ += (1.0/b->data()) * out->grad();
                b->grad_ += -1.0 * a->data() * pow(b->data(), -2.0) * out->grad();
            };

            return out;
        }
#endif

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> operator/(const Value<T>& a, N n) { return a / make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> operator/(N n, const Value<T>& a) { return make(n) / a; }

        // recip
        friend Value<T> recip(const Value<T>& a) {
            std::set<ptr> children = {a};
            double t = pow(a->data(), -1.0);
            auto out = make(t, children, "recip");

            //std::cerr << "Made recip node " << out << std::endl;

            out->backward_ = [=]() {
                a->grad_ += (-1.0 * pow(a->data(), -2.0)) * out->grad();
            };

            return out;
        }

        // exp
        friend Value<T> exp(const Value<T>& a) {
            std::set<ptr> children = {a};
            double t = exp(a->data());
            auto out = make(t, children, "exp");

            //std::cerr << "Made exp node " << out << std::endl;

            out->backward_ = [=]() {
                a->grad_ += out->data_ * out->grad_;
            };

            return out;
        }

        // pow
        friend Value<T> pow(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            double t = pow(a->data(), b->data());
            auto out = make(t, children, "pow");

            //std::cerr << "Made pow node " << out << std::endl;

            out->backward_ = [=]() {
                a->grad_ += (b->data() * pow(a->data(), (b->data()-1))) * out->grad();
            };

            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> pow(const Value<T>& a, N n) { return pow(a, make(n)); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend Value<T> pow(N n, const Value<T>& a) { return pow(make(n), a); }

        // tanh
        friend Value<T> tanh(const Value<T>& a) {
            std::set<ptr> children = {a};
            double x = a->data();
            double e2x = exp(2.0*x);
            double t = (e2x-1)/(e2x+1);
            auto out = make(t, children, "tanh");

            //std::cerr << "Made tanh node " << out << std::endl;

            out->backward_ = [=]() {
                double t = out->data_;
                a->grad_ += (1.0 - t*t) * out->grad_;
            };

            return out;
        }

    private:
        T data_;
        double grad_{0.0};
        std::string label_{};
        std::set<ptr> prev_{};
        std::string op_{""};

        std::function<void()> backward_{};
};

template <typename T, typename... Args>
static Value<T> leaf(const T& data, Args&&... args) {
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
