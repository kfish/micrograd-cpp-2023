#include <iostream>

#include "nn.h"

using namespace ai;

int main(int argc, char *argv[])
{
    std::array<double, 3> input = {0.1, 0.2, 0.3};

    Neuron<double, 3> n;

    std::cerr << n << std::endl;
    std::cerr << n(input) << std::endl;

    Layer<double, 3, 4> l;

    std::cerr << l << std::endl;
    std::cerr << PrettyArray(l(input)) << std::endl;
}

