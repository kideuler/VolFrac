#include "TorchWrapper.hpp"
#include <gtest/gtest.h>

TEST(TorchWrapperTest, Predict){
    TorchWrapper model("../models/VolFrac.pt", "../models/normalization.pt");
    double volfrac = model.Predict(0.5, 0.5, 0.0, 1.0, 0.001);

    double exact = 0.5;
    double percent_error = fabs(volfrac - exact) / exact * 100;
    std::cout << "Volume Fraction: " << volfrac << std::endl;
    std::cout << "Percent Error: " << percent_error << "%" << std::endl;
    ASSERT_TRUE(percent_error < 5.0);
}