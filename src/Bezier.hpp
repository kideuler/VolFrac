#ifndef __BEZIER_HPP__
#define __BEZIER_HPP__

#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>

/**
 * BezierCurve class for generating and evaluating Bezier curves
 * from a set of control points.
 */
class BezierCurve {
private:
    std::vector<std::array<double, 2>> controlPoints;
    
    /**
     * Calculate the binomial coefficient (n choose k)
     */
    unsigned long binomialCoeff(int n, int k) const;
    
    /**
     * Calculate the Bernstein polynomial value
     * B_{n,i}(t) = C(n,i) * t^i * (1-t)^(n-i)
     */
    double bernstein(int n, int i, double t) const;

public:
    /**
     * Constructor with initial control points
     */
    BezierCurve(const std::vector<std::array<double, 2>>& points);
    
    /**
     * Default constructor
     */
    BezierCurve();
    
    /**
     * Add a control point to the curve
     */
    void addControlPoint(const std::array<double, 2>& point);
    
    /**
     * Set all control points
     */
    void setControlPoints(const std::vector<std::array<double, 2>>& points);
    
    /**
     * Get all control points
     */
    const std::vector<std::array<double, 2>>& getControlPoints() const;
    
    /**
     * Clear all control points
     */
    void clear();
    
    /**
     * Evaluate the Bezier curve at parameter t where 0 <= t <= 1
     */
    std::array<double, 2> evaluate(double t) const;
    
    /**
     * Generate a series of points along the curve
     * @param numPoints The number of points to generate
     * @return Vector of points along the curve
     */
    std::vector<std::array<double, 2>> generateCurvePoints(int numPoints) const;
    
    /**
     * Calculate the derivative of the Bezier curve at parameter t
     */
    std::array<double, 2> derivative(double t) const;

    /**
     * Calculate the second derivative of the Bezier curve at parameter t
     */
    std::array<double, 2> secondDerivative(double t) const;
    
    /**
     * Calculate approximate arc length of the Bezier curve
     * using Gaussian quadrature with n sample points
     */
    double arcLength(int samples = 100) const;
};

#endif // __BEZIER_HPP__