#include <iostream>

#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    Value<double> a(2.0);
    Value<double> b(-3.0);
    Value<double> c(10.0);
    Value<double> expr1 = a*b;
    Value<double> expr = expr1 + c;
    //Value<double> expr = a*b + c;

    std::cout << Graph(expr) << std::endl;
}
