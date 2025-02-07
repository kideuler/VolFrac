#ifndef CIRCLEVOLFRAC_HPP
#define CIRCLEVOLFRAC_HPP

#include "IntervalTree.hpp"
#include <math.h>

double ComputeCircleBoxIntersection(const vertex &C, double r, double x_min, double x_max, double y_min, double y_max);

bool BoxCircleIntersection(double x, double y, double r, double x_min, double x_max, double y_min, double y_max);

#endif