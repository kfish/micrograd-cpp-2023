#pragma once

#include <iostream>
#include <set>

namespace ai {

template <typename T>
class Value {
    public:
        Value(const T& data)
            : data_(data)
        {}

        Value(const T& data, std::set<const Value<T>*>& children, char op='\0')
            : data_(data), prev_(children), op_(op)
        {}

        const T& data() const {
            return data_;
        }

        const std::set<const Value<T>*> children() const {
            return prev_;
        }

        char op() const {
            return op_;
        }

        Value<T> operator+(const Value<T>& other) const {
            std::set<const Value<T>*> children = {this, &other};
            return Value(data_ + other.data(), children, '+');
        }

        Value<T> operator-(const Value<T>& other) const {
            std::set<const Value<T>*> children = {this, &other};
            return Value(data_ - other.data(), children, '-');
        }

        Value<T> operator*(const Value<T>& other) const {
            std::set<const Value<T>*> children = {this, &other};
            return Value(data_ * other.data(), children, '*');
        }

        Value<T> operator/(const Value<T>& other) const {
            std::set<const Value<T>*> children = {this, &other};
            return Value(data_ / other.data(), children, '/');
        }
    private:
        T data_;
        std::set<const Value<T>*> prev_{};
        char op_{'\0'};
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const Value<T>& value) {
    return os << "Value(data=" << value.data() << ")";
}

}
