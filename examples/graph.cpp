#include <iostream>

#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = make_value(2.0, "a");
    auto b = make_value(-3.0, "b");
    auto c = make_value(10.0, "c");

    auto e = expr(a*b, "e");
    auto d = expr(e+c, "d");

    auto f = make_value(-2.0, "f");
    auto L = expr(d * f, "L");

    std::cout << Graph(L) << std::endl;
}
