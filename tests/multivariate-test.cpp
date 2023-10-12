#include <iostream>

#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = make_value(-2.0, "a");
    auto b = make_value(3.0, "b");

    auto d = expr(a*b, "d");
    auto e = expr(a+b, "e");
    auto f = expr(d*e, "f");

    backward(f);

    std::cout << Graph(f) << std::endl;
}
