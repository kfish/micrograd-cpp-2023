#pragma once

#include <iostream>
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
        RawValue(const T& data, const std::string& label)
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
            return ptr(new RawValue<T>(std::forward<Args>(args)...));
        }

        const T& data() const {
            return data_;
        }

        double grad() const {
            return grad_;
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

        friend Value<T> operator+(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() + b->data(), children, "+");
        }

        friend Value<T> operator-(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() - b->data(), children, "-");
        }

        friend Value<T> operator*(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() * b->data(), children, "*");
        }

        friend Value<T> operator/(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() / b->data(), children, "/");
        }

    private:
        T data_;
        double grad_{0.0};
        std::string label_{};
        std::set<ptr> prev_{};
        std::string op_{""};
};

template <typename T, typename... Args>
static Value<T> leaf(const T& data, Args&&... args) {
    return RawValue<T>::make(data, std::forward<Args>(args)...);
}

template <typename T>
static Value<T> expr(std::shared_ptr<RawValue<T>> unlabelled, const std::string& label) {
    return RawValue<T>::add_label(unlabelled, label);
}

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const RawValue<T>& value) {
    return os << "Value(data=" << value.data() << ")";
}

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<RawValue<T>>& value) {
    return os << "Value(data=" << value->data() << ")";
}

}
