#include <iostream>

#include "value.h"

using namespace ai;

int main(int argc, char *argv[])
{
    Value<double> x(2.0);

    std::cout << x << std::endl;
}
