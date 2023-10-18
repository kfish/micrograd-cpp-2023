#include <iostream>

#include "nn.h"
#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    // Define a neural net
    MLP1<double, 3, 4, 4, 1> n;

    std::array<double, 3> input = {{ 2.0, 3.0, -1.0 }};
    auto output = n(input);

    backward(output);

    std::cout << Graph(output) << std::endl;
}


