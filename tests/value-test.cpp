#include <iostream>

#include "value.h"

using namespace ai;

int main(int argc, char *argv[])
{
    Value<double> a(2.0);
    Value<double> b(-3.0);
    Value<double> c(10.0);

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << (a+b) << std::endl;
    std::cout << (a*b) << std::endl;

    std::cout << (a-b) << std::endl;
    std::cout << (a/b) << std::endl;

    std::cout << (a*b + c) << std::endl;
}
