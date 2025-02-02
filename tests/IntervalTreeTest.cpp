#include <IntervalTree.hpp>
#include <gtest/gtest.h>

TEST(IntervalTreeTest, ContructXTree) {
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
    IntervalTree<Axis::X> tree_X(segments, coordinates);
}

TEST(IntervalTreeTest, ContructYTree) {
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
    IntervalTree<Axis::Y> tree_Y(segments, coordinates);
}

TEST(IntervalTreeTest, QueryPoint) {
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

    vertex P = {0, 0};
    int count = tree.QueryPoint(P);
    ASSERT_EQ(count, 1);

    P = {0.5, 0.5};
    count = tree.QueryPoint(P);
    ASSERT_EQ(count, 1);

    P = {2, 0};
    count = tree.QueryPoint(P);
    ASSERT_EQ(count, 0);

    P = {-2, 0};
    count = tree.QueryPoint(P);
    ASSERT_EQ(count, 2);
}