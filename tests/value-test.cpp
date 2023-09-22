#include <iostream>

#include "value.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = leaf(2.0, "a");
    auto b = leaf(-3.0, "b");
    auto c = leaf(10.0, "c");

    std::cout << *a << std::endl;
    std::cout << b << std::endl;

    auto x = a + b;
    std::cout << x << std::endl;

    std::cout << a+b << std::endl;

    std::cout << (a*b) << std::endl;

    std::cout << (a-b) << std::endl;
    std::cout << (a/b) << std::endl;

    auto y = (a*b) + c;
    std::cout << y << std::endl;
    std::cout << (a*b + c) << std::endl;
}
