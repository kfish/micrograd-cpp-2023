#include <iostream>

#include "nn.h"

using namespace ai;

int main(int argc, char *argv[])
{
    Neuron<double, 2> n;

    std::cerr << n << std::endl;

    Layer<double, 2, 3> l;

    std::cerr << l << std::endl;
}

