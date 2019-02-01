#pragma once

#include <Eigen/Dense>

void poissonExample() {
    std::default_random_engine generator;
    std::poisson_distribution<int> distribution(1.1);
    auto poisson = [&] (int) {return distribution(generator);};

    RowVectorXi v = RowVectorXi::NullaryExpr(100, poisson );
    std::cout << v << "\n";
}
