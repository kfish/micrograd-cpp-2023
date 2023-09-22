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
        RawValue(const T& data)
            : data_(data)
        {}

#if 1
        RawValue(const T& data, std::set<ptr>& children, char op='\0')
            : data_(data), prev_(children), op_(op)
        {}
#endif

    public:
        template <typename... Args>
        static ptr make(Args&&... args) {
            return ptr(new RawValue<T>(std::forward<Args>(args)...));
        }

        const T& data() const {
            return data_;
        }

        const std::set<ptr> children() const {
            return prev_;
        }

        char op() const {
            return op_;
        }

        friend Value<T> operator+(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() + b->data(), children, '+');
        }

        friend Value<T> operator-(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() - b->data(), children, '-');
        }

        friend Value<T> operator*(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() * b->data(), children, '*');
        }

        friend Value<T> operator/(const Value<T>& a, const Value<T>& b) {
            std::set<ptr> children = {a, b};
            return make(a->data() / b->data(), children, '/');
        }
    private:
        T data_;
        std::set<ptr> prev_{};
        char op_{'\0'};
};

template <typename T, typename... Args>
static Value<T> make_value(const T& data, Args&&... args) {
    return RawValue<T>::make(data, std::forward<Args>(args)...);
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
