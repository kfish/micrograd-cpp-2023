#include <iostream>

#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = leaf(2.0, "a");
    auto b = leaf(-3.0, "b");
    auto c = leaf(10.0, "c");

    auto e = expr(a*b, "e");
    auto d = expr(e+c, "d");

    auto f = leaf(-2.0, "f");
    auto L = expr(d * f, "L");

    std::cout << Graph(L) << std::endl;
}
