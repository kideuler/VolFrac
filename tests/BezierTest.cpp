#include "gtest/gtest.h"
#include "Bezier.hpp"
#include <cmath>

// Helper function to compare points with tolerance
void ExpectPointsNear(const std::array<double, 2>& actual, 
                     const std::array<double, 2>& expected, 
                     double tolerance = 1e-10) {
    EXPECT_NEAR(actual[0], expected[0], tolerance);
    EXPECT_NEAR(actual[1], expected[1], tolerance);
}

// Test construction and basic operations
TEST(BezierTest, ConstructionAndBasicOps) {
    // Test default constructor
    BezierCurve emptyCurve;
    EXPECT_EQ(emptyCurve.getControlPoints().size(), 0);
    
    // Create points for a linear curve
    std::vector<std::array<double, 2>> linearPoints = {{0.0, 0.0}, {1.0, 1.0}};
    
    // Test constructor with points
    BezierCurve curve(linearPoints);
    EXPECT_EQ(curve.getControlPoints().size(), 2);
    
    // Test adding points
    emptyCurve.addControlPoint({1.0, 2.0});
    emptyCurve.addControlPoint({3.0, 4.0});
    EXPECT_EQ(emptyCurve.getControlPoints().size(), 2);
    EXPECT_EQ(emptyCurve.getControlPoints()[0][0], 1.0);
    EXPECT_EQ(emptyCurve.getControlPoints()[0][1], 2.0);
    
    // Test clearing points
    curve.clear();
    EXPECT_EQ(curve.getControlPoints().size(), 0);
    
    // Test exception for invalid number of points
    EXPECT_THROW(curve.setControlPoints({}), std::invalid_argument);
}

// Test evaluation of linear curve
TEST(BezierTest, LinearEvaluation) {
    // Create linear curve
    std::vector<std::array<double, 2>> linearPoints = {{0.0, 0.0}, {1.0, 1.0}};
    BezierCurve linearCurve(linearPoints);
    
    // Test endpoints
    ExpectPointsNear(linearCurve.evaluate(0.0), {0.0, 0.0});
    ExpectPointsNear(linearCurve.evaluate(1.0), {1.0, 1.0});
    
    // Test midpoint
    ExpectPointsNear(linearCurve.evaluate(0.5), {0.5, 0.5});
    
    // Test other points
    ExpectPointsNear(linearCurve.evaluate(0.25), {0.25, 0.25});
    ExpectPointsNear(linearCurve.evaluate(0.75), {0.75, 0.75});
}

// Test evaluation of quadratic curve
TEST(BezierTest, QuadraticEvaluation) {
    // Create quadratic curve (simple parabola)
    std::vector<std::array<double, 2>> quadraticPoints = {{0.0, 0.0}, {0.5, 1.0}, {1.0, 0.0}};
    BezierCurve quadraticCurve(quadraticPoints);
    
    // Test endpoints
    ExpectPointsNear(quadraticCurve.evaluate(0.0), {0.0, 0.0});
    ExpectPointsNear(quadraticCurve.evaluate(1.0), {1.0, 0.0});
    
    // Test midpoint (should be the peak of the parabola)
    ExpectPointsNear(quadraticCurve.evaluate(0.5), {0.5, 0.5});
    
    // Test other points
    // For quadratic Bezier: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    double t = 0.25;
    double x = (1-t)*(1-t)*0.0 + 2*(1-t)*t*0.5 + t*t*1.0;
    double y = (1-t)*(1-t)*0.0 + 2*(1-t)*t*1.0 + t*t*0.0;
    ExpectPointsNear(quadraticCurve.evaluate(0.25), {x, y});
}

// Test evaluation of cubic curve
TEST(BezierTest, CubicEvaluation) {
    // Create cubic curve (simple S-curve)
    std::vector<std::array<double, 2>> cubicPoints = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    BezierCurve cubicCurve(cubicPoints);
    
    // Test endpoints
    ExpectPointsNear(cubicCurve.evaluate(0.0), {0.0, 0.0});
    ExpectPointsNear(cubicCurve.evaluate(1.0), {1.0, 1.0});
    
    // Test other points
    // For cubic Bezier: 
    // B(t) = (1-t)^3 * P0 + 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2 + t^3 * P3
    double t = 0.5;
    double x = (1-t)*(1-t)*(1-t)*0.0 + 3*(1-t)*(1-t)*t*0.0 + 
               3*(1-t)*t*t*1.0 + t*t*t*1.0;
    double y = (1-t)*(1-t)*(1-t)*0.0 + 3*(1-t)*(1-t)*t*1.0 + 
               3*(1-t)*t*t*0.0 + t*t*t*1.0;
    ExpectPointsNear(cubicCurve.evaluate(0.5), {x, y}, 1e-10);
}

// Test generate curve points
TEST(BezierTest, GenerateCurvePoints) {
    // Create linear curve
    std::vector<std::array<double, 2>> linearPoints = {{0.0, 0.0}, {1.0, 1.0}};
    BezierCurve linearCurve(linearPoints);
    
    BezierCurve emptyCurve;
    
    // Test with 5 points
    auto points = linearCurve.generateCurvePoints(5);
    EXPECT_EQ(points.size(), 5);
    ExpectPointsNear(points[0], {0.0, 0.0});
    ExpectPointsNear(points[1], {0.25, 0.25});
    ExpectPointsNear(points[2], {0.5, 0.5});
    ExpectPointsNear(points[3], {0.75, 0.75});
    ExpectPointsNear(points[4], {1.0, 1.0});
    
    // Test with invalid number of points
    EXPECT_THROW(linearCurve.generateCurvePoints(1), std::invalid_argument);
    
    // Test with empty curve
    EXPECT_THROW(emptyCurve.generateCurvePoints(5), std::runtime_error);
}

// Test derivatives
TEST(BezierTest, FirstDerivative) {
    // Create curves
    std::vector<std::array<double, 2>> linearPoints = {{0.0, 0.0}, {1.0, 1.0}};
    std::vector<std::array<double, 2>> quadraticPoints = {{0.0, 0.0}, {0.5, 1.0}, {1.0, 0.0}};
    
    BezierCurve linearCurve(linearPoints);
    BezierCurve quadraticCurve(quadraticPoints);
    BezierCurve emptyCurve;
    
    // Linear curve derivative (constant)
    auto linearDeriv = linearCurve.derivative(0.5);
    ExpectPointsNear(linearDeriv, {1.0, 1.0});
    
    // Quadratic curve derivative (linear)
    // For a quadratic curve B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
    // The derivative is B'(t) = 2(P1-P0) + 2t*(P0-2P1+P2)
    auto quadDeriv = quadraticCurve.derivative(0.5);
    // At t=0.5, this should be 2(0.5-0) + 2(0.5)*(0-2*0.5+1) = 1.0, 0.0
    ExpectPointsNear(quadDeriv, {1.0, 0.0}, 1e-10);
    
    // Test with empty curve
    EXPECT_THROW(emptyCurve.derivative(0.5), std::runtime_error);
}

// Test second derivatives
TEST(BezierTest, SecondDerivative) {
    // Create curves
    std::vector<std::array<double, 2>> linearPoints = {{0.0, 0.0}, {1.0, 1.0}};
    std::vector<std::array<double, 2>> quadraticPoints = {{0.0, 0.0}, {0.5, 1.0}, {1.0, 0.0}};
    
    BezierCurve linearCurve(linearPoints);
    BezierCurve quadraticCurve(quadraticPoints);
    BezierCurve emptyCurve;
    
    // Quadratic curve second derivative (constant)
    // For a quadratic curve, second derivative is 2*(P0-2P1+P2)
    auto quadSecondDeriv = quadraticCurve.secondDerivative(0.5);
    // This should be 2*(0-2*1+1) = 2*(0-2) = 0.0, -4.0
    ExpectPointsNear(quadSecondDeriv, {0.0, -4.0}, 1e-10);
    
    // Test with curve that has too few points
    EXPECT_THROW(linearCurve.secondDerivative(0.5), std::runtime_error);
    
    // Test with empty curve
    EXPECT_THROW(emptyCurve.secondDerivative(0.5), std::runtime_error);
}

// Test arc length calculations
TEST(BezierTest, ArcLength) {
    // Create curves
    std::vector<std::array<double, 2>> linearPoints = {{0.0, 0.0}, {1.0, 1.0}};
    std::vector<std::array<double, 2>> circleQuadrant = {{1.0, 0.0}, {1.0, 0.55}, {0.55, 1.0}, {0.0, 1.0}};
    
    BezierCurve linearCurve(linearPoints);
    BezierCurve circleCurve(circleQuadrant);
    BezierCurve emptyCurve;
    
    // Linear curve (should be sqrt(2))
    double linearLength = linearCurve.arcLength(1000);
    EXPECT_NEAR(linearLength, std::sqrt(2.0), 1e-10);
    
    // Test circle quadrant (should be approximately PI/2)
    double circleLength = circleCurve.arcLength(1000);
    EXPECT_NEAR(circleLength, M_PI/2, 0.01); // Allow 1% error for approximation
    
    // Test with empty curve
    EXPECT_THROW(emptyCurve.arcLength(), std::runtime_error);
}

// Test parameter bounds handling
TEST(BezierTest, ParameterBounds) {
    // Create linear curve
    std::vector<std::array<double, 2>> linearPoints = {{0.0, 0.0}, {1.0, 1.0}};
    BezierCurve linearCurve(linearPoints);
    
    // Test t < 0
    ExpectPointsNear(linearCurve.evaluate(-0.5), linearCurve.evaluate(0.0));
    
    // Test t > 1
    ExpectPointsNear(linearCurve.evaluate(1.5), linearCurve.evaluate(1.0));
    
    // Test derivatives with out-of-bounds t
    ExpectPointsNear(linearCurve.derivative(-0.5), linearCurve.derivative(0.0));
    ExpectPointsNear(linearCurve.derivative(1.5), linearCurve.derivative(1.0));
}