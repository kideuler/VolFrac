#include "Grid.hpp"
#include <gtest/gtest.h>

TEST(VolumeFractionTest, ComputeVolumeFractionsCircle){

    // Define the number of segments
    const int num_segments = 150;
    const double pi = 3.14159265358979323846;

    // Create coordinates for the unit circle
    coords coordinates;
    for (int i = 0; i < num_segments; ++i) {
        double angle = 2 * pi * i / num_segments;
        coordinates.push_back({cos(angle), sin(angle)});
    }

    // Create segment ids for the unit circle
    segment_ids segments;
    for (int i = 0; i < num_segments; ++i) {
        segments.push_back({i, (i + 1) % num_segments});
    }

    // Create the IntervalTree
    IntervalTree<Axis::Y> tree(segments, coordinates);

    BBox box{-1.01, 1.01, -1.01, 1.01};
    Grid grid(box, 10, 10);
    grid.AddShape(tree);
    grid.ComputeVolumeFractions(2);
    double total_volume = grid.ComputeTotalVolume();
    double percent_error = 100 * fabs(total_volume - M_PI) / M_PI;
    std::cout << "Percent Error: " << percent_error << std::endl;
    ASSERT_TRUE(percent_error < 5);

    Grid grid2(box, 100, 100);
    grid2.AddShape(tree);
    grid2.ComputeVolumeFractions(20);
    total_volume = grid2.ComputeTotalVolume();
    percent_error = 100 * fabs(total_volume - M_PI) / M_PI;
    std::cout << "Percent Error: " << percent_error << std::endl;
    ASSERT_TRUE(percent_error < 0.05);

}

TEST(VolumeFractionTest, ComputeVolumeFractionsFlower){

    // Define the number of segments
    const int num_segments = 1000;
    const double pi = 3.14159265358979323846;

    // Create coordinates for the unit circle
    coords coordinates;
    for (int i = 0; i < num_segments; ++i) {
        double t = 2 * pi * i / num_segments;
        double x = ((0.25 + 0.1*sin(5*t))*cos(t))/3.0 + 0.5;
        double y = ((0.25 + 0.1*sin(5*t))*sin(t))/3.0 + 0.5;
        coordinates.push_back({x,y});
    }

    // Create segment ids for the unit circle
    segment_ids segments;
    for (int i = 0; i < num_segments; ++i) {
        segments.push_back({i, (i + 1) % num_segments});
    }

    // Create the IntervalTree
    IntervalTree<Axis::Y> tree(segments, coordinates);

    double exact = 0.0075*M_PI;

    BBox box{-0.01, 1.01, -0.01, 1.01};
    Grid grid(box, 50, 50);
    grid.AddShape(tree);
    grid.ComputeVolumeFractions(20);
    double total_volume = grid.ComputeTotalVolume();
    double percent_error = 100 * fabs(total_volume - exact) / exact;
    std::cout << "percent_error" << percent_error << std::endl;
    ASSERT_TRUE(percent_error < 5);
}

TEST(VolumeFractionTest, ComputeVolumeFractionsCardioid){

    // Define the number of segments
    const int num_segments = 1000;
    const double pi = 3.14159265358979323846;
    const double a = 0.1;

    // Create coordinates for the unit circle
    coords coordinates;
    for (int i = 0; i < num_segments; ++i) {
        double t = 2 * pi * i / num_segments;
        double x = 2*a*(1-cos(t))*cos(t)/2+ 0.5;
        double y = 2*a*(1-cos(t))*sin(t)/2+ 0.5;
        coordinates.push_back({x,y});
    }

    // Create segment ids for the unit circle
    segment_ids segments;
    for (int i = 0; i < num_segments; ++i) {
        segments.push_back({i, (i + 1) % num_segments});
    }

    // Create the IntervalTree
    IntervalTree<Axis::Y> tree(segments, coordinates);

    double exact = 0.015*M_PI;

    BBox box{-0.01, 1.01, -0.01, 1.01};
    Grid grid(box, 50, 50);
    grid.AddShape(tree);
    grid.ComputeVolumeFractions(20);
    double total_volume = grid.ComputeTotalVolume();
    double percent_error = 100 * fabs(total_volume - exact) / exact;
    std::cout << "percent_error" << percent_error << std::endl;
    ASSERT_TRUE(percent_error < 5);
}