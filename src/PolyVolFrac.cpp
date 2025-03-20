#include "PolyVolFrac.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

/**
 * 2D point representation
 */
struct Point {
    double x, y;
    
    Point() : x(0), y(0) {}
    Point(double _x, double _y) : x(_x), y(_y) {}
};

/**
 * Convert from normal vector and point to plane equation coefficients
 * 
 * Plane equation: nx*x + ny*y + d = 0
 * For a plane with normal (nx, ny) passing through point (px, py):
 * d = -(nx*px + ny*py)
 * 
 * @param normal_x X component of normal vector
 * @param normal_y Y component of normal vector
 * @param point_x X coordinate of point on plane
 * @param point_y Y coordinate of point on plane
 * @param plane Output plane coefficients [nx, ny, d]
 */
void pointNormalToPlane(double normal_x, double normal_y, double point_x, double point_y, double plane[3]) {
    // Normalize the normal vector
    double length = std::sqrt(normal_x*normal_x + normal_y*normal_y);
    if (length > 1e-10) {
        normal_x /= length;
        normal_y /= length;
    }
    
    plane[0] = normal_x;
    plane[1] = normal_y;
    plane[2] = -(normal_x*point_x + normal_y*point_y);
}

/**
 * Determines if a point is on the negative side of a plane
 * 
 * @param p The point to test
 * @param plane The plane coefficients [nx, ny, d]
 * @return true if point is on negative side (kept side)
 */
bool isInside(const Point& p, const double plane[3]) {
    // Point is inside if nx*x + ny*y + d < 0
    return plane[0] * p.x + plane[1] * p.y + plane[2] < 0;
}

/**
 * Computes the intersection point of a line segment with a plane
 * 
 * @param p1 First endpoint of line segment
 * @param p2 Second endpoint of line segment
 * @param plane Plane coefficients [nx, ny, d]
 * @return Intersection point
 */
Point computeIntersection(const Point& p1, const Point& p2, const double plane[3]) {
    double nx = plane[0];
    double ny = plane[1];
    double d = plane[2];
    
    double denominator = nx * (p2.x - p1.x) + ny * (p2.y - p1.y);
    
    // Handle near-parallel cases
    if (std::abs(denominator) < 1e-10) {
        // Return midpoint as fallback
        return Point((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5);
    }
    
    // Parameter along line segment where intersection occurs
    double t = -(nx * p1.x + ny * p1.y + d) / denominator;
    
    // Clamp t to [0,1] for numerical stability
    t = std::max(0.0, std::min(1.0, t));
    
    // Compute intersection point
    return Point(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y));
}

/**
 * Clips a polygon against a plane using Sutherland-Hodgman algorithm
 * 
 * @param inputPolygon The input polygon vertices
 * @param plane The clipping plane
 * @return The clipped polygon
 */
std::vector<Point> clipPolygonAgainstPlane(const std::vector<Point>& inputPolygon, const double plane[3]) {
    std::vector<Point> outputPolygon;
    
    if (inputPolygon.empty()) {
        return outputPolygon;
    }
    
    Point s = inputPolygon.back();
    bool s_inside = isInside(s, plane);
    
    for (const Point& e : inputPolygon) {
        bool e_inside = isInside(e, plane);
        
        // If ending point is inside clipping plane
        if (e_inside) {
            // Add the intersection point if starting point was outside
            if (!s_inside) {
                outputPolygon.push_back(computeIntersection(s, e, plane));
            }
            // Always add ending point if it's inside
            outputPolygon.push_back(e);
        } 
        // If ending point is outside but starting was inside
        else if (s_inside) {
            // Add only the intersection point
            outputPolygon.push_back(computeIntersection(s, e, plane));
        }
        
        // Update for next edge
        s = e;
        s_inside = e_inside;
    }
    
    return outputPolygon;
}

/**
 * Calculate area of a polygon using the shoelace formula
 * 
 * @param polygon The polygon vertices
 * @return The area of the polygon
 */
double calculatePolygonArea(const std::vector<Point>& polygon) {
    if (polygon.size() < 3) {
        return 0.0;
    }
    
    double area = 0.0;
    
    for (size_t i = 0; i < polygon.size(); i++) {
        size_t j = (i + 1) % polygon.size();
        area += polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
    }
    
    return std::abs(area) / 2.0;
}

/**
 * Calculate the area of a box after clipping with multiple planes
 * 
 * @param x_min Minimum x-coordinate of box
 * @param x_max Maximum x-coordinate of box
 * @param y_min Minimum y-coordinate of box
 * @param y_max Maximum y-coordinate of box
 * @param planes Array of plane coefficients [nx, ny, d]
 * @param nplanes Number of planes
 * @return Area of the clipped box
 */
double PlaneBoxIntersection(double x_min, double x_max, double y_min, double y_max, double planes[][3], int nplanes) {
    // Initial polygon is the box
    std::vector<Point> polygon = {
        Point(x_min, y_min),
        Point(x_max, y_min),
        Point(x_max, y_max),
        Point(x_min, y_max)
    };
    
    // Apply Sutherland-Hodgman algorithm for each clipping plane
    for (int i = 0; i < nplanes; i++) {
        polygon = clipPolygonAgainstPlane(polygon, planes[i]);
        
        if (polygon.empty()) {
            return 0.0;
        }
    }
    
    return calculatePolygonArea(polygon);
}