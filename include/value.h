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

        Value(const T& data, std::set<Value<T>>& children, char op=' ')
            : data_(data), prev_(children), op_(op)
        {}

        const T& data() const {
            return data_;
        }

        Value<T> operator+(const Value<T>& other) const {
            return Value(data_ + other.data());
        }

        Value<T> operator-(const Value<T>& other) const {
            return Value(data_ - other.data());
        }

        Value<T> operator*(const Value<T>& other) const {
            return Value(data_ * other.data());
        }

        Value<T> operator/(const Value<T>& other) const {
            return Value(data_ / other.data());
        }
    private:
        T data_;
        std::set<Value<T>> prev_{};
        char op_{' '};
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const Value<T>& value) {
    return os << "Value(data=" << value.data() << ")";
}

}
