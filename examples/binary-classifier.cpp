#include <iostream>

#include "randomdata.h"
#include "loss.h"

#include "nn.h"

using namespace ai;

int main(int argc, char *argv[])
{
    // Define a neural net
    MLP<double, 3, 4, 4, 1> n;

    std::cerr << n << std::endl;

    auto input0 = value_array<double, 3>({ 2.0, 3.0, -1.0 });
    auto gt0 = 1.0;

    std::cout << PrettyArray(input0) << std::endl;

    // Result of a single input
    Value<double> pred0 = n(input0)[0];

    //std::cout << PrettyArray(n(input0)) << std::endl;
    std::cout << "Pred0:\t" << pred0 << std::endl;

    // Loss of that
    Value<double> loss0 = mse_loss(pred0, gt0);
    std::cout << "Loss0:\t" << loss0 << std::endl;

    std::array<std::array<double,3>, 4> input = {{
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    }};

    // Result of many inputs
    //auto each0 = [&](const auto & x) { return n(x)[0]; };
    //auto pred = array_transform(each0, input);
    //auto pred = n(input);

    auto pred = std::array<Value<double>, 4>();
    auto each0 = [&](const auto & x) { return n(x)[0]; };
    std::transform(input.begin(), input.end(), pred.begin(), each0);

    for (int i : {0, 1, 2, 3}) {
        //    std::cout << PrettyArray(n(input[i])) << std::endl;
        //std::cout << PrettyArray(pred[i]) << std::endl;
    }

    std::array<double, 4> y = {1.0, -1.0, -1.0, 1.0};
    std::cout << PrettyArray(y) << std::endl;

    // Avg loss of those
    //double loss = mse_loss(pred, y);
    Value<double> loss = mse_loss(pred, y);
    std::cout << "Loss:\t" << loss << std::endl;

    // Run backprop
    //
    // Adjust gradients
    //
    // Recalc loss
    //
    //
    // (play with strategies: all together, biases first, sense first etc

}

