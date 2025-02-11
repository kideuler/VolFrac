#include "Grid.hpp"
#include <gtest/gtest.h>


TEST(GridTest, ConstructGrid){
    BBox box{0.0, 1.0, 0.0, 1.0};
    Grid grid(box, 100, 100);
}

TEST(GridTest, AddShape){
    BBox box{0.0, 1.0, 0.0, 1.0};
    Grid grid(box, 10, 10);

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
    auto tree = std::make_unique<IntervalTree<Axis::Y>>(segments, coordinates);
    grid.AddShape(std::move(tree));
}