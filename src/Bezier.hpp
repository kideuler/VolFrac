#ifndef __BEZIER_HPP__
#define __BEZIER_HPP__

#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>

/**
 * @class BezierCurve
 * @brief Class for generating and evaluating parametric Bezier curves in 2D space
 *
 * This class implements a Bezier curve defined by a set of control points.
 * A Bezier curve is a parametric curve that uses the Bernstein polynomials
 * as a basis. The resulting curve starts at the first control point and 
 * ends at the last control point, with the other control points defining
 * the shape and path of the curve.
 */
class BezierCurve {
private:
    /** @brief Vector containing the control points that define the curve */
    std::vector<std::array<double, 2>> controlPoints;
    
    /**
     * @brief Calculate the binomial coefficient (n choose k)
     * 
     * @param n Total number of items
     * @param k Number of items to choose
     * @return The binomial coefficient value
     */
    unsigned long binomialCoeff(int n, int k) const;
    
    /**
     * @brief Calculate the Bernstein polynomial value
     * 
     * Computes B_{n,i}(t) = C(n,i) * t^i * (1-t)^(n-i) where C(n,i) is the
     * binomial coefficient.
     * 
     * @param n Degree of the Bernstein polynomial
     * @param i Index of the Bernstein polynomial
     * @param t Parameter value in range [0,1]
     * @return Value of the Bernstein polynomial at t
     */
    double bernstein(int n, int i, double t) const;

public:
    /**
     * @brief Constructs a Bezier curve with the given control points
     * 
     * @param points Vector of 2D points that define the curve's control polygon
     */
    BezierCurve(const std::vector<std::array<double, 2>>& points);
    
    /**
     * @brief Default constructor that creates an empty Bezier curve
     */
    BezierCurve();
    
    /**
     * @brief Adds a new control point to the curve
     * 
     * Adding a control point changes the shape of the curve. The curve's
     * degree increases by one for each control point added.
     * 
     * @param point The 2D control point to add
     */
    void addControlPoint(const std::array<double, 2>& point);
    
    /**
     * @brief Sets all control points for the curve, replacing any existing points
     * 
     * @param points Vector of 2D points that define the new control polygon
     */
    void setControlPoints(const std::vector<std::array<double, 2>>& points);
    
    /**
     * @brief Gets the current control points of the Bezier curve
     * 
     * @return Constant reference to the vector of control points
     */
    const std::vector<std::array<double, 2>>& getControlPoints() const;
    
    /**
     * @brief Removes all control points, resulting in an undefined curve
     */
    void clear();
    
    /**
     * @brief Evaluates the position of the Bezier curve at parameter t
     * 
     * @param t Parameter value in range [0,1] where 0 represents the start point
     *        and 1 represents the end point of the curve
     * @return The 2D point on the curve at parameter t
     * @throws std::runtime_error if the curve has no control points
     */
    std::array<double, 2> evaluate(double t) const;
    
    /**
     * @brief Generates a discretized representation of the curve with equally spaced parameters
     * 
     * @param numPoints The number of points to generate along the curve
     * @return Vector of 2D points representing the curve
     */
    std::vector<std::array<double, 2>> generateCurvePoints(int numPoints) const;
    
    /**
     * @brief Calculates the first derivative of the Bezier curve at parameter t
     * 
     * The derivative represents the tangent vector to the curve at the given parameter.
     * The magnitude of this vector indicates the rate of change of position with respect to t.
     * 
     * @param t Parameter value in range [0,1]
     * @return The derivative vector at parameter t
     * @throws std::runtime_error if the curve has fewer than 2 control points
     */
    std::array<double, 2> derivative(double t) const;

    /**
     * @brief Calculates the second derivative of the Bezier curve at parameter t
     * 
     * The second derivative represents the curvature characteristics of the curve.
     * 
     * @param t Parameter value in range [0,1]
     * @return The second derivative vector at parameter t
     * @throws std::runtime_error if the curve has fewer than 3 control points
     */
    std::array<double, 2> secondDerivative(double t) const;
    
    /**
     * @brief Calculates the approximate arc length of the Bezier curve
     * 
     * Uses numerical integration (Gaussian quadrature) to approximate the 
     * arc length of the curve.
     * 
     * @param samples Number of sample points to use in the approximation (higher gives better accuracy)
     * @return The approximate arc length of the curve
     */
    double arcLength(int samples = 100) const;
};

#endif // __BEZIER_HPP__