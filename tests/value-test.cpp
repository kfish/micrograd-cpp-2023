#include <iostream>

#include "value.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = make_value(2.0);
    auto b = make_value(-3.0);
    auto c = make_value(10.0);

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
