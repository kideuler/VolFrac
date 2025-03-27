#ifndef __POLYVOLFRAC_HPP__
#define __POLYVOLFRAC_HPP__

/**
 * @file PolyVolFrac.hpp
 * @brief Functions for computing polygon-based volume fractions using plane clipping
 *
 * This file provides utilities for calculating the volume (area in 2D) of regions
 * formed by intersections of planes with rectangular boxes. These functions are used
 * for high-accuracy volume fraction computations in cut cells.
 */

// ASSUME SIDE WITH NORMAL GETS CUT

/**
 * @brief Computes the area of intersection between a box and a set of half-planes
 *
 * Calculates the area of a polygon formed by clipping a rectangular box with
 * a set of planes. The function implements the Sutherland-Hodgman polygon clipping
 * algorithm to successively clip the box against each plane. Each plane is defined
 * by the equation ax + by + c = 0, where the normal vector (a,b) points toward
 * the region to be kept.
 *
 * @param x_min Minimum x-coordinate of the box
 * @param x_max Maximum x-coordinate of the box
 * @param y_min Minimum y-coordinate of the box
 * @param y_max Maximum y-coordinate of the box
 * @param planes Array of plane coefficients, where each plane is [a, b, c] for equation ax + by + c = 0
 * @param nplanes Number of planes to clip against
 * @return The area of the polygon formed by the intersection of the box and all planes
 *
 * @note The function assumes the side of the plane with the normal vector is the side that gets cut.
 *       If the box is entirely outside any plane, the function returns 0.
 */
double PlaneBoxIntersection(double x_min, double x_max, double y_min, double y_max, double planes[][3], int nplanes);

#endif // __POLYVOLFRAC_HPP__