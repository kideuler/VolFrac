#include "Bezier.hpp"

unsigned long BezierCurve::binomialCoeff(int n, int k) const {
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k; // Symmetry property
    
    unsigned long c = 1;
    for (int i = 0; i < k; ++i) {
        c = c * (n - i) / (i + 1);
    }
    
    return c;
}

double BezierCurve::bernstein(int n, int i, double t) const {
    return binomialCoeff(n, i) * std::pow(t, i) * std::pow(1.0 - t, n - i);
}

BezierCurve::BezierCurve(const std::vector<std::array<double, 2>>& points) 
    : controlPoints(points) {
    if (points.size() < 2) {
        throw std::invalid_argument("Bezier curve requires at least 2 control points");
    }
}

BezierCurve::BezierCurve() {}

void BezierCurve::addControlPoint(const std::array<double, 2>& point) {
    controlPoints.push_back(point);
}

void BezierCurve::setControlPoints(const std::vector<std::array<double, 2>>& points) {
    if (points.size() < 2) {
        throw std::invalid_argument("Bezier curve requires at least 2 control points");
    }
    controlPoints = points;
}

const std::vector<std::array<double, 2>>& BezierCurve::getControlPoints() const {
    return controlPoints;
}

void BezierCurve::clear() {
    controlPoints.clear();
}

std::array<double, 2> BezierCurve::evaluate(double t) const {
    if (controlPoints.size() < 2) {
        throw std::runtime_error("Cannot evaluate Bezier curve with fewer than 2 control points");
    }
    
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    
    int n = controlPoints.size() - 1; // Degree of the curve
    std::array<double, 2> result = {0.0, 0.0};
    
    // Calculate the point using the Bernstein polynomial form
    for (int i = 0; i <= n; ++i) {
        double b = bernstein(n, i, t);
        result[0] += controlPoints[i][0] * b;
        result[1] += controlPoints[i][1] * b;
    }
    
    return result;
}

std::vector<std::array<double, 2>> BezierCurve::generateCurvePoints(int numPoints) const {
    if (numPoints < 2) {
        throw std::invalid_argument("Number of points must be at least 2");
    }
    
    std::vector<std::array<double, 2>> curvePoints;
    curvePoints.reserve(numPoints);
    
    for (int i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / (numPoints - 1);
        curvePoints.push_back(evaluate(t));
    }
    
    return curvePoints;
}

std::array<double, 2> BezierCurve::derivative(double t) const {
    if (controlPoints.size() < 2) {
        throw std::runtime_error("Cannot evaluate derivative with fewer than 2 control points");
    }
    
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    
    int n = controlPoints.size() - 1; // Degree of the curve
    std::array<double, 2> result = {0.0, 0.0};
    
    // Calculate the derivative using the formula:
    // B'(t) = n * sum_{i=0}^{n-1} [b_{n-1,i}(t) * (P_{i+1} - P_i)]
    for (int i = 0; i < n; ++i) {
        double b = bernstein(n - 1, i, t);
        result[0] += n * b * (controlPoints[i+1][0] - controlPoints[i][0]);
        result[1] += n * b * (controlPoints[i+1][1] - controlPoints[i][1]);
    }
    
    return result;
}

std::array<double, 2> BezierCurve::secondDerivative(double t) const {
    if (controlPoints.size() < 3) {
        throw std::runtime_error("Cannot evaluate second derivative with fewer than 3 control points");
    }
    
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    
    int n = controlPoints.size() - 1; // Degree of the curve
    std::array<double, 2> result = {0.0, 0.0};
    
    // Calculate the second derivative using the formula:
    // B''(t) = n(n-1) * sum_{i=0}^{n-2} [b_{n-2,i}(t) * (P_{i+2} - 2P_{i+1} + P_i)]
    for (int i = 0; i <= n - 2; ++i) {
        double b = bernstein(n - 2, i, t);
        double factor = n * (n - 1) * b;
        
        // Calculate P_{i+2} - 2P_{i+1} + P_i for x and y components
        double x_term = controlPoints[i+2][0] - 2*controlPoints[i+1][0] + controlPoints[i][0];
        double y_term = controlPoints[i+2][1] - 2*controlPoints[i+1][1] + controlPoints[i][1];
        
        result[0] += factor * x_term;
        result[1] += factor * y_term;
    }
    
    return result;
}

double BezierCurve::arcLength(int samples) const {
    if (controlPoints.size() < 2) {
        throw std::runtime_error("Cannot calculate arc length with fewer than 2 control points");
    }
    
    double length = 0.0;
    double dt = 1.0 / samples;
    
    for (int i = 0; i < samples; ++i) {
        double t1 = i * dt;
        double t2 = (i + 1) * dt;
        
        std::array<double, 2> d1 = derivative(t1);
        std::array<double, 2> d2 = derivative(t2);
        
        // Speed at t1 and t2
        double speed1 = std::sqrt(d1[0]*d1[0] + d1[1]*d1[1]);
        double speed2 = std::sqrt(d2[0]*d2[0] + d2[1]*d2[1]);
        
        // Trapezoidal rule for integration
        length += 0.5 * (speed1 + speed2) * dt;
    }
    
    return length;
}