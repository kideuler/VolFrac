#include "CircleVolFrac.hpp"
#include <assert.h>


// This code is from https://stackoverflow.com/questions/622287/area-of-intersection-between-circle-and-rectangle modified to use doubles
static inline double section(double h, double r = 1) // returns the positive root of intersection of line y = h with circle centered at the origin and radius r
{
    assert(r >= 0); // assume r is positive, leads to some simplifications in the formula below (can factor out r from the square root)
    return (h < r)? sqrt(r * r - h * h) : 0; // http://www.wolframalpha.com/input/?i=r+*+sin%28acos%28x+%2F+r%29%29+%3D+h
}

static inline double g(double x, double h, double r = 1) // indefinite integral of circle segment
{
    return .5f * (sqrt(1 - x * x / (r * r)) * x * r + r * r * asin(x / r) - 2 * h * x); // http://www.wolframalpha.com/input/?i=r+*+sin%28acos%28x+%2F+r%29%29+-+h
}

static inline double area(double x0, double x1, double h, double r) // area of intersection of an infinitely tall box with left edge at x0, right edge at x1, bottom edge at h and top edge at infinity, with circle centered at the origin with radius r
{
    if(x0 > x1)
        std::swap(x0, x1); // this must be sorted otherwise we get negative area
    double s = section(h, r);
    return g(max(-s, min(s, x1)), h, r) - g(max(-s, min(s, x0)), h, r); // integrate the area
}

static double area(double x0, double x1, double y0, double y1, double r) // area of the intersection of a finite box with a circle centered at the origin with radius r
{
    if(y0 > y1)
        std::swap(y0, y1); // this will simplify the reasoning
    if(y0 < 0) {
        if(y1 < 0)
            return area(x0, x1, -y0, -y1, r); // the box is completely under, just flip it above and try again
        else
            return area(x0, x1, 0, -y0, r) + area(x0, x1, 0, y1, r); // the box is both above and below, divide it to two boxes and go again
    } else {
        assert(y1 >= 0); // y0 >= 0, which means that y1 >= 0 also (y1 >= y0) because of the swap at the beginning
        return area(x0, x1, y0, r) - area(x0, x1, y1, r); // area of the lower box minus area of the higher box
    }
}

double ComputeCircleBoxIntersection(const vertex &C, double r, double x_min, double x_max, double y_min, double y_max){
    double x0 = x_min-C[0];
    double x1 = x_max-C[0];
    double y0 = y_min-C[1];
    double y1 = y_max-C[1];

    return area(x0, x1, y0, y1, r);
}

// from https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
bool BoxCircleIntersection(double x, double y, double r, double x_min, double x_max, double y_min, double y_max){
    double dx = x_max - x_min;
    double dy = y_max - y_min;

    double circle_distance_x = fabs(x - x_min);
    double circle_distance_y = fabs(y - y_min);

    if (circle_distance_x > (dx/2 + r)) { return false; }
    if (circle_distance_y > (dy/2 + r)) { return false; }

    if (circle_distance_x <= (dx/2)) { return true; }
    if (circle_distance_y <= (dy/2)) { return true; }

    double corner_distance_sq = (circle_distance_x - dx/2)*(circle_distance_x - dx/2) + (circle_distance_y - dy/2)*(circle_distance_y - dy/2);

    return (corner_distance_sq <= (r*r));
}