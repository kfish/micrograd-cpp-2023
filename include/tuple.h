#pragma once

#include <tuple>

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

