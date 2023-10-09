#include <iostream>

#include "backprop.h"
#include "nn.h"
#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    // Define a neural net
    MLP1<double, 3, 4, 4, 1> n;

    std::cerr << n << std::endl;

    std::array<std::array<double,3>, 4> input = {{
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    }};

    std::array<double, 4> y = {1.0, -1.0, -1.0, 1.0};
    std::cerr << "y (gt):\t" << PrettyArray(y) << std::endl;

    // Run backprop
    double learning_rate = 0.01;

    auto backprop = BackProp<double, 4, std::array<double, 3>>(n, "loss.tsv");
    double loss = backprop(input, y, learning_rate, 1000, true);
}

