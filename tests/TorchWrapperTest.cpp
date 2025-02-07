#include "TorchWrapper.hpp"
#include <gtest/gtest.h>
#include "Grid.hpp"

TEST(TorchWrapperTest, Predict){
    TorchWrapper model("../../models/VolFrac.pt", "../../models/normalization.pt");
    double k = 0.001;
    vertex C{0.5, 0.5};
    double volfrac = model.Predict(C[0], C[1], 0.0, 1.0, k);
    
    vertex P{C[0], C[1]+1.0/k};
    double exact = ComputeCircleBoxIntersection(P, 1.0/k, 0.0,1.0,0.0,1.0);
    double percent_error = fabs(volfrac - exact) / exact * 100;
    std::cout << "Volume Fraction: " << volfrac << " Exact: " << exact << std::endl;
    std::cout << "Percent Error: " << percent_error << "%" << std::endl;
    ASSERT_TRUE(percent_error < 5.0);
    
}

TEST(TorchWrapperTest, Circle){
    const int num_segments = 1000;
    const double pi = 3.14159265358979323846;

    coords coordinates;
    KDTree<5> tree;

    for (int i = 0; i < num_segments; ++i) {
        double angle = 2 * pi * i / num_segments;
        double x = 0.5*cos(angle)+0.5;
        double y = 0.5*sin(angle)+0.5;
        vertex P{x, y};
        coordinates.push_back(P);

        double dx = -0.5*sin(angle);
        double dy = 0.5*cos(angle);
        double nrm = sqrt(dx*dx + dy*dy);
        dx /= nrm;
        dy /= nrm;

        double dxx = -0.5*cos(angle);
        double dyy = -0.5*sin(angle);
        double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
        double arr[5] = {dx, dy, dxx, dyy, k};

        tree.Insert(P, arr);
    }

    BBox box{-0.01, 1.01, -0.01, 1.01};
    Grid grid(box, 300, 300);
    grid.AddTree(tree);

    TorchWrapper model("../../models/VolFrac.pt", "../../models/normalization.pt");
    grid.model = &model;

    grid.ComputeVolumeFractionsAI();

    double exact = pi/4.0;
    double total_volume = grid.ComputeTotalVolume();
    double percent_error = 100 * fabs(total_volume - exact) / exact;
    std::cout << "Total Volume: " << total_volume << std::endl;
    std::cout << "Percent Error: " << percent_error << std::endl;
    ASSERT_TRUE(percent_error < 5.0);
}