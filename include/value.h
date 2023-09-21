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
    private:
        T data_;
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const Value<T>& value) {
    return os << value.data();
}

}
