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

    // A set of training inputs
    std::array<std::array<double,3>, 4> input = {{
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    }};

    // Corresponding ground truth values for these inputs
    std::array<double, 4> y = {1.0, -1.0, -1.0, 1.0};
    std::cerr << "y (gt):\t" << PrettyArray(y) << std::endl;

    double learning_rate = 0.9;

    auto backprop = BackProp<double, std::array<double, 3>, 4>(n, "loss.tsv");

    // Run backprop for 20 iterations, verbose=true
    double loss = backprop(input, y, learning_rate, 20, true);
}

