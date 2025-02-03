#include "KDTree.hpp"
#include "gtest/gtest.h"

TEST(KDTreeTest, Insert) {
    // Define the number of segments
    const int num_segments = 150;
    const double pi = 3.14159265358979323846;

    // Create coordinates for the unit circle
    coords coordinates;
    for (int i = 0; i < num_segments; ++i) {
        double angle = 2 * pi * i / num_segments;
        coordinates.push_back({cos(angle), sin(angle)});
    }

    KDTree<1> tree;
    for (int i = 0; i < num_segments; ++i) {
        double arr[1] = {0.0};
        tree.Insert(coordinates[i], arr);
    }
}