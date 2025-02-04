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

TEST(KDTreeTest, Search){
    // Define the number of segments
    const int num_segments = 400;
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

    // check  closest point to 0,-1
    vertex P{0.0, -0.9};
    double best_radius = 1.0;
    vertex best_point{0.0, 0.0};
    double best_data[1] = {0.0};
    tree.Search(P, best_point, best_data);
    double dis = (best_point[0] - 0) * (best_point[0] - 0) + (best_point[1] - -1) * (best_point[1] - -1);
    ASSERT_NEAR(dis, 0.0, 1e-14);

    // check  closest point to 1,0
    P = {0.9, 0.0};
    best_radius = 1.0;
    best_point = {0.0, 0.0};
    best_data[0] = 0.0;
    tree.Search(P, best_point, best_data);
    dis = (best_point[0] - 1) * (best_point[0] - 1) + (best_point[1] - 0) * (best_point[1] - 0);
    ASSERT_NEAR(dis, 0.0, 1e-14);

    // check  closest point to 0,1
    P = {0.0, 0.9};
    best_radius = 1.0;
    best_point = {0.0, 0.0};
    best_data[0] = 0.0;
    tree.Search(P, best_point, best_data);
    dis = (best_point[0] - 0) * (best_point[0] - 0) + (best_point[1] - 1) * (best_point[1] - 1);
    ASSERT_NEAR(dis, 0.0, 1e-14);

    // check  closest point to -1,0
    P = {-0.9, 0.0};
    best_radius = 1.0;
    best_point = {0.0, 0.0};
    best_data[0] = 0.0;
    tree.Search(P, best_point, best_data);
    dis = (best_point[0] - -1) * (best_point[0] - -1) + (best_point[1] - 0) * (best_point[1] - 0);
    ASSERT_NEAR(dis, 0.0, 1e-14);

}

TEST(KDTreeTest, EllipseClosestPointTest){
    // Define the number of segments
    const int num_segments = 1000;
    const double pi = 3.14159265358979323846;

    KDTree<4> tree;
    coords coordinates;
    // Create coordinates for the ellipse
    for (int i = 0; i < num_segments; ++i) {
        double t = 2 * pi * i / num_segments;
        double x = 0.2*cos(t)+0.5;
        double y = 0.1*sin(t)+0.5;
        vertex P{x, y};
        coordinates.push_back(P);

        double dx = -0.2*sin(t);
        double dy = 0.1*cos(t);

        double dxx = -0.2*cos(t);
        double dyy = -0.1*sin(t);
        double arr[4] = {dx, dy, dxx, dyy};

        tree.Insert(P, arr);
    }


    // use random points to search and check that they are the same to loopibng through all points
    for (int v = 0; v < 50; v++){
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        vertex P{x, y};

        vertex best_point;
        double best_data[4];
        tree.Search(P, best_point, best_data);

        vertex best_point_loop;
        double best_dist = 1e34;
        for (int i = 0; i < num_segments; ++i) {
            double dx = -0.2*sin(2*pi*i/num_segments);
            double dy = 0.1*cos(2*pi*i/num_segments);
            double dxx = -0.2*cos(2*pi*i/num_segments);
            double dyy = -0.1*sin(2*pi*i/num_segments);
            double dis = (coordinates[i][0] - P[0]) * (coordinates[i][0] - P[0]) + (coordinates[i][1] - P[1]) * (coordinates[i][1] - P[1]);
            if (dis < best_dist){
                best_point_loop = coordinates[i];
                best_dist = dis;
            }
        }

        ASSERT_NEAR(best_point[0], best_point_loop[0], 1e-15);
        ASSERT_NEAR(best_point[1], best_point_loop[1], 1e-15);
    }
}