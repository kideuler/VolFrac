#include "../models/Model.hpp"
#include "CircleVolFrac.hpp"
#include "gtest/gtest.h"

TEST(ModelTest, LoadModel) {
    Model model("model.dat");
    model.PrintModel();
}

TEST(ModelTest, Activation) {
    Model model("model.dat");
    Vector vec(5);
    vec(0) = 1.0;
    vec(1) = -1.0;
    vec(2) = 0.0;
    vec(3) = 2.0;
    vec(4) = -2.0;
    model.applyActivation(vec);
    EXPECT_DOUBLE_EQ(vec(0), 1.0);
    EXPECT_DOUBLE_EQ(vec(1), 0.0);
    EXPECT_DOUBLE_EQ(vec(2), 0.0);
    EXPECT_DOUBLE_EQ(vec(3), 2.0);
    EXPECT_DOUBLE_EQ(vec(4), 0.0);
}

TEST(ModelTest, Prediction) {
    Model model("model.dat");
    
    // test against circle and a unit square
    double k = 0.00001;
    double data[5] = {0.5, 0.5, 0.0, 1.0, k};
    double volfrac_ai = model.Predict(data);

    double r = 1/k;
    double volfrac = ComputeCircleBoxIntersection({0.5+r,0.5}, r, 0.0, 1.0, 0.0, 1.0);

    std::cout << "AI: " << volfrac_ai << std::endl;
    std::cout << "Circle: " << volfrac << std::endl;

    double percent_error = fabs(volfrac - volfrac_ai) / volfrac * 100;
    std::cout << "Percent Error: " << percent_error << std::endl;
    EXPECT_NEAR(percent_error, 0.0, 5.0);
}