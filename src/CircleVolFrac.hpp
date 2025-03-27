#ifndef CIRCLEVOLFRAC_HPP
#define CIRCLEVOLFRAC_HPP

#include "IntervalTree.hpp"
#include <math.h>

/**
 * @file CircleVolFrac.hpp
 * @brief Functions for computing circle-box intersections and volume fractions
 *
 * This file contains utilities for calculating the intersection area between 
 * circles and rectangular boxes, used primarily for volume fraction computations
 * in the computational grid.
 */

/**
 * @brief Computes the intersection area between a circle and a rectangular box
 *
 * Calculates the exact area of the region formed by the intersection of a circle 
 * with center C and radius r, and a rectangular box defined by its min/max coordinates.
 *
 * @param C The center coordinates of the circle as a vertex (x,y)
 * @param r The radius of the circle
 * @param x_min Minimum x-coordinate of the box
 * @param x_max Maximum x-coordinate of the box
 * @param y_min Minimum y-coordinate of the box
 * @param y_max Maximum y-coordinate of the box
 * @return The area of the intersection between the circle and the box
 */
double ComputeCircleBoxIntersection(const vertex &C, double r, double x_min, double x_max, double y_min, double y_max);

/**
 * @brief Determines if a circle and a rectangular box intersect
 *
 * Tests whether a circle with center (x,y) and radius r intersects with a 
 * rectangular box defined by its min/max coordinates. This is a fast check
 * that doesn't compute the actual intersection area.
 *
 * @param x The x-coordinate of the circle's center
 * @param y The y-coordinate of the circle's center
 * @param r The radius of the circle
 * @param x_min Minimum x-coordinate of the box
 * @param x_max Maximum x-coordinate of the box
 * @param y_min Minimum y-coordinate of the box
 * @param y_max Maximum y-coordinate of the box
 * @return True if the circle and box intersect, false otherwise
 */
bool BoxCircleIntersection(double x, double y, double r, double x_min, double x_max, double y_min, double y_max);

#endif // CIRCLEVOLFRAC_HPP