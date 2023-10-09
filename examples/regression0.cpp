#include <iostream>

#include "backprop.h"
#include "nn.h"
#include "graph.h"

using namespace ai;

class Regression0 {
    public:
        Regression0()
            : weight_(randomValue<double>())
        {}

        Value<double> weight() const {
            return weight_;
        }

        Value<double> operator()(const Value<double>& x) const {
            return weight_ * x;
        }

        Value<double> operator()(const double& x) const {
            return this->operator()(leaf(x));
        }

        void adjust(const double& learning_rate) {
            weight_->adjust(learning_rate);
        }

    private:
        Value<double> weight_;
};

static inline std::ostream& operator<<(std::ostream& os, const Regression0& r)
{
    return os << "Regression0{weight=" << r.weight() << "}";
}

int main(int argc, char *argv[])
{
    Regression0 n;

    std::cerr << n << std::endl;

    std::array<double, 4> input = {
        {-7.0, -3.0, 1.0, 4.0},
    };

    std::array<double, 4> y = {-21.0, -9.0, 3.0, 12.0};
    std::cerr << "y (gt):\t" << PrettyArray(y) << std::endl;

    // Run backprop
    double learning_rate = 0.01;

    auto backprop = BackProp<double, 4, double>(n, "loss.tsv");

    double loss = backprop(input, y, learning_rate, 40, true);

    std::cout << Graph(backprop.loss_function()(input, y)) << std::endl;
}

