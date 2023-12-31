#include <iostream>

#include "value.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = make_value(2.0, "a");
    auto b = make_value(-3.0, "b");
    auto c = make_value(10.0, "c");

    std::cout << *a << std::endl;
    std::cout << b << std::endl;

    auto x = a + b;
    std::cout << x << std::endl;

    std::cout << a+b << std::endl;

    std::cout << a+7.0 << std::endl;

    std::cout << 7.0+b << std::endl;

    std::cout << a+7.0+b << std::endl;

    std::cout << a+b+c << std::endl;

    std::cout << (a*b) << std::endl;

    std::cout << (a-b) << std::endl;

    auto minus1 = make_value(-1.0, "-1.0");
    auto nb = expr(b * minus1, "nb");
    std::cout << (a + nb) << std::endl;

    auto g = expr(a/b, "a/b");;
    std::cout << g << std::endl;

    auto tg = expr(tanh(g), "tanh(g)");
    std::cout << tg << std::endl;
    backward(tg);

    auto br = expr(recip(b), "br");;
    std::cout << br << std::endl;

    auto f = expr(a * br, "f");
    std::cout << f << std::endl;

    auto tt = expr(tanh(f), "tanh(f)");;
    std::cout << tt << std::endl;

    backward(tt);

    auto y = (a*b) + c;
    std::cout << y << std::endl;
    std::cout << (a*b + c) << std::endl;
}
