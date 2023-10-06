#pragma once

#include <tuple>

template <std::size_t N, size_t... Ns>
void print_args_impl(std::ostream& os) {
    os << N;
    ((os << ',' << Ns), ...);
}

template <std::size_t... Ns>
void print_args(std::ostream& os) {
    os << "<";
    if constexpr (sizeof...(Ns) > 0) {
        print_args_impl<Ns...>(os);
    }
    os << ">";
}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), std::ostream&>::type
print_tuple(std::ostream& os, const std::tuple<Tp...>& t)
{
    return os;
}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), std::ostream&>::type
print_tuple(std::ostream& os, const std::tuple<Tp...>& t)
{
    os << std::get<I>(t);
    if (I + 1 != sizeof...(Tp)) os << ", ";
    return print_tuple<I + 1, Tp...>(os, t);
}

