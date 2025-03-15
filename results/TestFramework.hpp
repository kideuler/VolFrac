#ifndef __TEST_FRAMEWORK_HPP__
#define __TEST_FRAMEWORK_HPP__

#include "Grid.hpp"
#include <iomanip>

Grid CreateGrid(BBox box, int nx, int ny, int shape_type, int nsegs = 10000) {
    Grid grid(box, nx, ny);

    const double pi = 3.14159265358979323846;
    const int num_segments = nsegs;
    if (shape_type>2){
        return grid;
    }

    coords coordinates;
    KDTree<5> tree;
    segment_ids segments;
    std::vector<std::array<double,5>> data;

    switch (shape_type){
        case 0: // Ellipse
            for (int i = 0; i < num_segments; ++i) {
                double angle = 2 * pi * i / num_segments;
                double x = 0.2*cos(angle)+0.5;
                double y = 0.1*sin(angle)+0.5;
                vertex P{x, y};
                coordinates.push_back(P);

                double dx = -0.2*sin(angle);
                double dy = 0.1*cos(angle);

                double dxx = -0.2*cos(angle);
                double dyy = -0.1*sin(angle);
                double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
                double nrm = sqrt(dx*dx + dy*dy);
                dx /= nrm;
                dy /= nrm;
                double arr[5] = {dx, dy, dxx, dyy, k};
                std::array<double,5> arr2 = {dx, dy, dxx, dyy, k};
                data.push_back(arr2);


                tree.Insert(P, arr);
                segments.push_back({i, (i + 1) % num_segments});
            }
            break;

        case 1: // Petaled Flower
            for (int i = 0; i < num_segments; ++i) {
                double t = 2 * pi * i / num_segments;
                double x = ((0.25 + 0.1*sin(5*t))*cos(t))/3.0 + 0.5;
                double y = ((0.25 + 0.1*sin(5*t))*sin(t))/3.0 + 0.5;
                vertex P{x, y};
                coordinates.push_back(P);

                double dx = -(0.1*sin(5*t) + 0.25)*sin(t)/3 + 0.166666666666667*cos(t)*cos(5*t);
                double dy =(0.1*sin(5*t) + 0.25)*cos(t)/3 + 0.166666666666667*sin(t)*cos(5*t);
                
                double dxx = -(0.1*sin(5*t) + 0.25)*cos(t)/3 - 0.333333333333333*sin(t)*cos(5*t) - 0.833333333333333*sin(5*t)*cos(t);
                double dyy = -(0.1*sin(5*t) + 0.25)*sin(t)/3 - 0.833333333333333*sin(t)*sin(5*t) + 0.333333333333333*cos(t)*cos(5*t);
                double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
                double nrm = sqrt(dx*dx + dy*dy);
                dx /= nrm;
                dy /= nrm;

                double arr[5] = {dx, dy, dxx, dyy, k};
                std::array<double,5> arr2 = {dx, dy, dxx, dyy, k};
                data.push_back(arr2);

                tree.Insert(P, arr);
                segments.push_back({i, (i + 1) % num_segments});
            }
            break;
        case 2: // Extreme Flower
            for (int i = 0; i < num_segments; ++i) {
                double t = 2 * pi * i / num_segments;
                double x = (0.25 + 0.15*sin(20*t))*cos(t)/3.0 + 0.5;
                double y = (0.25 + 0.1*sin(20*t))*sin(t)/3.0 + 0.5;
                vertex P{x, y};
                coordinates.push_back(P);

                double dx = -0.333333333333333*(0.15*sin(20*t) + 0.25)*sin(t) + 1.0*cos(t)*cos(20*t);
                double dy =0.333333333333333*(0.1*sin(20*t) + 0.25)*cos(t) + 0.666666666666667*sin(t)*cos(20*t);

                double dxx = -0.333333333333333*(0.15*sin(20*t) + 0.25)*cos(t) - 2.0*sin(t)*cos(20*t) - 20.0*sin(20*t)*cos(t);
                double dyy = -0.333333333333333*(0.1*sin(20*t) + 0.25)*sin(t) - 13.3333333333333*sin(t)*sin(20*t) + 1.33333333333333*cos(t)*cos(20*t);
                double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
                double nrm = sqrt(dx*dx + dy*dy);
                dx /= nrm;
                dy /= nrm;
                double arr[5] = {dx, dy, dxx, dyy, k};
                std::array<double,5> arr2 = {dx, dy, dxx, dyy, k};
                data.push_back(arr2);

                tree.Insert(P, arr);
                segments.push_back({i, (i + 1) % num_segments});
            }
            break;
    }

    auto shape = std::make_unique<IntervalTree<Axis::Y>>(segments, coordinates, data);
    grid.AddShape(std::move(shape));
    grid.AddTree(tree);
    
    return grid;
}

#endif