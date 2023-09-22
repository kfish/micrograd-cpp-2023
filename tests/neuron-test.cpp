#include <iostream>

#include "graph.h"

using namespace ai;

int main(int argc, char *argv[])
{
    // Inputs x1, x2
    auto x1 = leaf(2.0, "x1");
    auto x2 = leaf(0.0, "x2");

    // Weights w1, w2
    auto w1 = leaf(-3.0, "w1");
    auto w2 = leaf(1.0, "w2");

    // Bias of the neuron
    auto b = leaf(6.8813735870195432, "b");

    auto x1w1 = expr(x1*w1, "x1*w1");
    auto x2w2 = expr(x2*w2, "x2*w2");

    auto x1w1x2w2 = expr(x1w1 + x2w2, "x1w1+x2w2");
    auto n = expr(x1w1x2w2 + b, "n");

    auto o = expr(tanh(n), "o");

    backward(o);

    std::cout << Graph(o) << std::endl;
}
