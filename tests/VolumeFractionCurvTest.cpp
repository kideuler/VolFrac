#include "Grid.hpp"
#include <gtest/gtest.h>

TEST(VolumeFractionCurveTest, ComputeVolumeFractionsCircle){

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

    // Create segment ids for the unit circle
    segment_ids segments;
    for (int i = 0; i < num_segments; ++i) {
        segments.push_back({i, (i + 1) % num_segments});
    }

    BBox box{-0.01, 1.01, -0.01, 1.01};
    Grid grid(box, 100, 100);
    grid.AddTree(tree);
    auto shape = std::make_unique<IntervalTree<Axis::Y>>(segments, coordinates);
    grid.AddShape(std::move(shape));

    grid.ComputeVolumeFractionsCurv();

    double exact = pi/4.0;
    double total_volume = grid.ComputeTotalVolume();
    double percent_error = 100 * fabs(total_volume - exact) / exact;
    std::cout << "Total Volume: " << total_volume << std::endl;
    std::cout << "Percent Error: " << percent_error << std::endl;
    ASSERT_TRUE(percent_error < 0.05);
}


TEST(VolumeFractionCurveTest, ComputeVolumeFractionsEllipse){

    const int num_segments = 1000;
    const double pi = 3.14159265358979323846;

    coords coordinates;
    KDTree<5> tree;

    for (int i = 0; i < num_segments; ++i) {
        double angle = 2 * pi * i / num_segments;
        double x = 0.2*cos(angle)+0.5;
        double y = 0.1*sin(angle)+0.5;
        vertex P{x, y};
        coordinates.push_back(P);

        double dx = -0.2*sin(angle);
        double dy = 0.1*cos(angle);
        double nrm = sqrt(dx*dx + dy*dy);
        dx /= nrm;
        dy /= nrm;

        double dxx = -0.2*cos(angle);
        double dyy = -0.1*sin(angle);
        double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
        double arr[5] = {dx, dy, dxx, dyy, k};

        tree.Insert(P, arr);
    }

    // Create segment ids for the unit circle
    segment_ids segments;
    for (int i = 0; i < num_segments; ++i) {
        segments.push_back({i, (i + 1) % num_segments});
    }

    BBox box{-0.01, 1.01, -0.01, 1.01};
    Grid grid(box, 100, 100);
    grid.AddTree(tree);
    auto shape = std::make_unique<IntervalTree<Axis::Y>>(segments, coordinates);
    grid.AddShape(std::move(shape));

    grid.ComputeVolumeFractionsCurv();

    double exact = pi*0.02;
    double total_volume = grid.ComputeTotalVolume();
    double percent_error = 100 * fabs(total_volume - exact) / exact;
    std::cout << "Total Volume: " << total_volume << std::endl;
    std::cout << "Percent Error: " << percent_error << std::endl;
    ASSERT_TRUE(percent_error < 0.05);
}

TEST(VolumeFractionCurveTest, ComputeVolumeFractionsFlower){

    const int num_segments = 3000;
    const double pi = 3.14159265358979323846;

    coords coordinates;
    KDTree<5> tree;

    for (int i = 0; i < num_segments; ++i) {
        double t = 2 * pi * i / num_segments;
        double x = ((0.25 + 0.1*sin(5*t))*cos(t))/3.0 + 0.5;
        double y = ((0.25 + 0.1*sin(5*t))*sin(t))/3.0 + 0.5;
        vertex P{x, y};
        coordinates.push_back(P);

        double dx = -(0.1*sin(5*t) + 0.25)*sin(t)/3 + 0.166666666666667*cos(t)*cos(5*t);
        double dy =(0.1*sin(5*t) + 0.25)*cos(t)/3 + 0.166666666666667*sin(t)*cos(5*t);
        double nrm = sqrt(dx*dx + dy*dy);
        dx /= nrm;
        dy /= nrm;

        double dxx = -(0.1*sin(5*t) + 0.25)*cos(t)/3 - 0.333333333333333*sin(t)*cos(5*t) - 0.833333333333333*sin(5*t)*cos(t);
        double dyy = -(0.1*sin(5*t) + 0.25)*sin(t)/3 - 0.833333333333333*sin(t)*sin(5*t) + 0.333333333333333*cos(t)*cos(5*t);
        double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
        double arr[5] = {dx, dy, dxx, dyy, k};

        tree.Insert(P, arr);
    }

    // Create segment ids for the unit circle
    segment_ids segments;
    for (int i = 0; i < num_segments; ++i) {
        segments.push_back({i, (i + 1) % num_segments});
    }
    BBox box{-0.01, 1.01, -0.01, 1.01};
    Grid grid(box, 130, 200);
    grid.AddTree(tree);
    auto shape = std::make_unique<IntervalTree<Axis::Y>>(segments, coordinates);
    grid.AddShape(std::move(shape));

    grid.ComputeVolumeFractionsCurv();

    double exact = 0.0075*M_PI;
    double total_volume = grid.ComputeTotalVolume();
    double percent_error = 100 * fabs(total_volume - exact) / exact;
    std::cout << "Total Volume: " << total_volume << std::endl;
    std::cout << "Percent Error: " << percent_error << std::endl;
    ASSERT_TRUE(percent_error < 0.05);
}