#include "CircleVolFrac.hpp"
#include <assert.h>


// ...existing code...
__attribute__((always_inline)) static inline double section(double h, double r = 1) {
    // Use builtin_expect to help branch prediction
    if (__builtin_expect(h < r, 1))
        return sqrt(r * r - h * h);
    return 0.0;
}
// ...existing code...
__attribute__((always_inline)) static inline double g(double x, double h, double r = 1) {
    double rr = r * r;
    return 0.5 * (sqrt(1.0 - (x * x) / rr) * x * r + rr * asin(x / r) - 2.0 * h * x);
}
// ...existing code...
__attribute__((always_inline)) static inline double area(double x0, double x1, double h, double r) {
    if (__builtin_expect(x0 > x1, 0))
        std::swap(x0, x1);
    double s = section(h, r);
    double left  = std::max(-s, std::min(s, x0));
    double right = std::max(-s, std::min(s, x1));
    return g(right, h, r) - g(left, h, r);
}
// ...existing code...

__attribute__((always_inline)) static inline double area(double x0, double x1, double y0, double y1, double r)
{
    // Help the compiler predict branches
    if (__builtin_expect(y0 > y1, 0))
        std::swap(y0, y1);

    if (__builtin_expect(y0 < 0, 1)) {
        if (__builtin_expect(y1 < 0, 1))
            return area(x0, x1, -y0, -y1, r);
        else
            return area(x0, x1, 0, -y0, r) + area(x0, x1, 0, y1, r);
    } else {
        return area(x0, x1, y0, r) - area(x0, x1, y1, r);
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