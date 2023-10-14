#include <iostream>

#include "value.h"
#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = make_value(-4.0, "a");
    auto b = make_value(2.0, "b");

    auto c = expr(a + b, "c");;
    auto d = a * b + pow(b, 3);

    c += c + 1;

    std::cout << Graph(c) << std::endl;
}
