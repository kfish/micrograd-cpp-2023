#include <iostream>

#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    auto a = make_value(2.0);
    auto b = make_value(-3.0);
    auto c = make_value(10.0);

    //Value<double> expr1 = a*b;
    //Value<double> expr = expr1 + c;
    auto expr = a*b + c;

    std::cout << Graph(expr) << std::endl;
}
