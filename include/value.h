#pragma once

#include <iostream>

namespace ai {

template <typename T>
class Value {
    public:
        Value(const T& data)
            : data_(data)
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
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const Value<T>& value) {
    return os << "Value(data=" << value.data() << ")";
}

}
