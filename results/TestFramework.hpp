#ifndef __TEST_FRAMEWORK_HPP__
#define __TEST_FRAMEWORK_HPP__

#include "Grid.hpp"
#include <iomanip>

const double pi = 3.14159265358979323846;

std::array<double,7> Ghost(double t) {
    
    std::array<double, 7> data;
    data[0] =  ((t >= 0 && t < 1.0) ? (
        0.5*cos(M_PI*t)
     )
     : ((t >= 1.0 && t < 2.0) ? (
        0.25*pow(t, 2) - 0.5*t - 0.25
     )
     : ((t >= 2.0 && t < 3.0) ? (
        1.25*t - 2.75
     )
     : ((t >= 3.0 && t < 4.0) ? (
        0.5*pow(t, 2) - 4.0*t + 8.5
     )
     : ((t >= 4.0 && t < 5.0) ? (
        0.10000000000000001*cos(2*M_PI*t) - 0.20000000000000001
     )
     : ((t >= 5.0 && t < 6.0) ? (
        0.10000000000000001*cos(2*M_PI*t) + 0.20000000000000001
     )
     : (
        NAN
     )))))));
     
     data[1] =  ((t >= 0 && t < 1.0) ? (
        0.5*sin(M_PI*t) + 1.0
     )
     : ((t >= 1.0 && t < 2.0) ? (
        2.0 - 1.0*t
     )
     : ((t >= 2.0 && t < 3.0) ? (
        0.050000000000000003*sin(6*M_PI*t)
     )
     : ((t >= 3.0 && t < 4.0) ? (
        1.0*t - 3.0
     )
     : (((t >= 4.0 || t > 5.0) && (t >= 4.0 || t < 6.0) && (t <= 5.0 || t < 6.0)) ? (
        0.20000000000000001*sin(2*M_PI*t) + 1.0
     )
     : (
        NAN
     ))))));
     
     data[2] =  ((t >= 0 && t < 1.0) ? (
        -0.5*M_PI*sin(M_PI*t)
     )
     : ((t >= 1.0 && t < 2.0) ? (
        0.5*t - 0.5
     )
     : ((t >= 2.0 && t < 3.0) ? (
        1.25
     )
     : ((t >= 3.0 && t < 4.0) ? (
        1.0*t - 4.0
     )
     : (((t >= 4.0 || t > 5.0) && (t >= 4.0 || t < 6.0) && (t <= 5.0 || t < 6.0)) ? (
        -0.20000000000000001*M_PI*sin(2*M_PI*t)
     )
     : (
        NAN
     ))))));
     
     data[3] =  ((t >= 0 && t < 1.0) ? (
        0.5*M_PI*cos(M_PI*t)
     )
     : ((t >= 1.0 && t < 2.0) ? (
        -1.0
     )
     : ((t >= 2.0 && t < 3.0) ? (
        0.30000000000000004*M_PI*cos(6*M_PI*t)
     )
     : ((t >= 3.0 && t < 4.0) ? (
        1.0
     )
     : (((t >= 4.0 || t > 5.0) && (t >= 4.0 || t < 6.0) && (t <= 5.0 || t < 6.0)) ? (
        0.40000000000000002*M_PI*cos(2*M_PI*t)
     )
     : (
        NAN
     ))))));

     data[4] =  ((t >= 0 && t < 1.0) ? (
        -0.5*pow(M_PI, 2)*cos(M_PI*t)
     )
     : ((t >= 1.0 && t < 2.0) ? (
        0.5
     )
     : ((t >= 2.0 && t < 3.0) ? (
        0
     )
     : ((t >= 3.0 && t < 4.0) ? (
        1.0
     )
     : (((t >= 4.0 || t > 5.0) && (t >= 4.0 || t < 6.0) && (t <= 5.0 || t < 6.0)) ? (
        -0.40000000000000002*pow(M_PI, 2)*cos(2*M_PI*t)
     )
     : (
        NAN
     ))))));
     
     data[5] =  ((t >= 0 && t < 1.0) ? (
        -0.5*pow(M_PI, 2)*sin(M_PI*t)
     )
     : ((t >= 1.0 && t < 2.0) ? (
        0
     )
     : ((t >= 2.0 && t < 3.0) ? (
        -1.8000000000000003*pow(M_PI, 2)*sin(6*M_PI*t)
     )
     : ((t >= 3.0 && t < 4.0) ? (
        0
     )
     : (((t >= 4.0 || t > 5.0) && (t >= 4.0 || t < 6.0) && (t <= 5.0 || t < 6.0)) ? (
        -0.80000000000000004*pow(M_PI, 2)*sin(2*M_PI*t)
     )
     : (
        NAN
     ))))));

    return data;
}

static inline vertex normalize_and_orient(std::array<double,7> data) {
    double x = data[0];
    double y = data[1];
    double dx = data[2];
    double dy = data[3];
    double dxx = data[4];
    double dyy = data[5];
    double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
    double nrm = sqrt(dx*dx + dy*dy);
    dx /= nrm;
    dy /= nrm;
    if (k < 0.0) {
        dx = -dx;
        dy = -dy;
    }
    return {dx, dy};
}

// point_x, point_y, d1x, d1y, d2x, d2y
std::vector<std::array<double,6>> GhostDC() {
    std::vector<std::array<double,6>> data;
    // discontinuity at 4 and 0
    std::array<double,7> d1, d2;
    vertex n1, n2;
    double x,y,dx,dy,dxx,dyy,k,nrm;
    d1 = Ghost(3.999);
    d2 = Ghost(0.00);
    n1 = normalize_and_orient(d1);
    n2 = normalize_and_orient(d2);
    data.push_back({d2[0], d2[1], n1[0], n1[1], n2[0], n2[1]});

    d1 = Ghost(0.999);
    d2 = Ghost(1.00);
    n1 = normalize_and_orient(d1);
    n2 = normalize_and_orient(d2);
    data.push_back({d2[0], d2[1], n1[0], n1[1], n2[0], n2[1]});

    d1 = Ghost(1.999);
    d2 = Ghost(2.00);
    n1 = normalize_and_orient(d1);
    n2 = normalize_and_orient(d2);
    data.push_back({d2[0], d2[1], n1[0], n1[1], n2[0], n2[1]});

    d1 = Ghost(2.999);
    d2 = Ghost(3.00);
    n1 = normalize_and_orient(d1);
    n2 = normalize_and_orient(d2);
    data.push_back({d2[0], d2[1], n1[0], n1[1], n2[0], n2[1]});
    return data;
}

Grid CreateGrid(BBox box, int nx, int ny, int shape_type, int nsegs = 10000) {
    Grid grid(box, nx, ny);

    const int num_segments = nsegs;
    if (shape_type>3){
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

                // if curvature is negative flip normal and second derivative
                if (k < 0.0) {
                    dx = -dx;
                    dy = -dy;
                    dxx = -dxx;
                    dyy = -dyy;
                }

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

                // if curvature is negative flip normal and second derivative
                if (k < 0.0) {
                    dx = -dx;
                    dy = -dy;
                    dxx = -dxx;
                    dyy = -dyy;
                }

                double arr[5] = {dx, dy, dxx, dyy, k};
                std::array<double,5> arr2 = {dx, dy, dxx, dyy, k};
                data.push_back(arr2);

                tree.Insert(P, arr);
                segments.push_back({i, (i + 1) % num_segments});
            }
            break;
        case 3: // Ghost
            for (int i = 0; i < num_segments; ++i) {
                double t = 4.0 * double(i) / double(num_segments);
                std::array<double,7> data1 = Ghost(t);
                double x = data1[0];
                double y = data1[1];
                vertex P{x, y};
                coordinates.push_back(P);

                double dx = data1[2];
                double dy = data1[3];

                double dxx = data1[4];
                double dyy = data1[5];
                
                double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
                double nrm = sqrt(dx*dx + dy*dy);
                dx /= nrm;
                dy /= nrm;

                if (k < 0.0) {
                    dx = -dx;
                    dy = -dy;
                    dxx = -dxx;
                    dyy = -dyy;
                }

                double arr[5] = {dx, dy, dxx, dyy, k};
                std::array<double,5> arr2 = {dx, dy, dxx, dyy, k};
                data.push_back(arr2);

                tree.Insert(P, arr);
                segments.push_back({i, (i + 1) % num_segments});
            }
            
            // left eye
            for (int i = num_segments; i < num_segments+num_segments/4; ++i) {
                double t = 1.0 * double(i-num_segments) / double(num_segments/4) + 4.0;
                std::array<double,7> data1 = Ghost(t);
                double x = data1[0];
                double y = data1[1];
                vertex P{x, y};
                coordinates.push_back(P);

                double dx = -data1[2];
                double dy = -data1[3];

                double dxx = -data1[4];
                double dyy = -data1[5];

                double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
                double nrm = sqrt(dx*dx + dy*dy);
                dx /= nrm;
                dy /= nrm;

                double arr[5] = {dx, dy, dxx, dyy, k};
                std::array<double,5> arr2 = {dx, dy, dxx, dyy, k};
                data.push_back(arr2);

                tree.Insert(P, arr);
                segments.push_back({i, (i + 1) % (num_segments+num_segments/4)});
            }

            // Right eye
            for (int i = num_segments+num_segments/4; i < num_segments+num_segments/2; ++i) {
                double t = 1.0 * double(i-(num_segments+num_segments/4)) / double(num_segments/4) + 5.0;
                std::array<double,7> data1 = Ghost(t);
                double x = data1[0];
                double y = data1[1];
                vertex P{x, y};
                coordinates.push_back(P);

                double dx = -data1[2];
                double dy = -data1[3];

                double dxx = -data1[4];
                double dyy = -data1[5];
                
                double k = (dx*dyy - dy*dxx) / pow(dx*dx + dy*dy, 1.5);
                double nrm = sqrt(dx*dx + dy*dy);
                dx /= nrm;
                dy /= nrm;

                double arr[5] = {dx, dy, dxx, dyy, k};
                std::array<double,5> arr2 = {dx, dy, dxx, dyy, k};
                data.push_back(arr2);

                tree.Insert(P, arr);
                segments.push_back({i, (i + 1) % (num_segments+num_segments/2)});
            }
            grid.discontinuities = GhostDC();
            break;
    }

    auto shape = std::make_unique<IntervalTree<Axis::Y>>(segments, coordinates, data);
    grid.AddShape(std::move(shape));
    grid.AddTree(tree);
    
    return grid;
}

#endif