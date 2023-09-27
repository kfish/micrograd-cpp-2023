#include <iostream>

#include "randomdata.h"
#include "nn.h"

using namespace ai;

int main(int argc, char *argv[])
{
    std::array<Value<double>, 3> input = {leaf(0.1), leaf(0.2), leaf(0.3)};

    Neuron<double, 3> n;

    std::cerr << n << std::endl;
    std::cerr << n(input) << std::endl;

    Layer<double, 3, 4> l;

    std::cerr << l << std::endl;
    std::cerr << PrettyArray(l(input)) << std::endl;

    MLP<double, 3, 4, 2> mlp;
    std::cerr << mlp << std::endl;
    std::cerr << PrettyArray(mlp(input)) << std::endl;
}

