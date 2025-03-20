#include "PolyVolFrac.hpp"
#include "gtest/gtest.h"
#include <cmath>

// Tolerance for floating point comparisons
constexpr double EPSILON = 1e-10;

// Test when no planes are provided (should return full box area)
TEST(PolyVolFracTest, NoPlanes) {
    double planes[0][3] = {};
    double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 0);
    EXPECT_DOUBLE_EQ(1.0, area);
}

// Test when a plane cuts the box in half diagonally
TEST(PolyVolFracTest, DiagonalCut) {
    // Plane with normal (1,1) through point (0.5,0.5)
    // Equation: x + y - 1 = 0
    double planes[1][3] = {
        {1.0, 1.0, -1.0}
    };
    double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 1);
    EXPECT_NEAR(0.5, area, EPSILON);
}

// Test with horizontal and vertical cuts
TEST(PolyVolFracTest, RectangularCut) {
    // Cut from right at x = 0.7
    // Cut from top at y = 0.8
    double planes[2][3] = {
        {1.0, 0.0, -0.7},  // x - 0.7 = 0 => x = 0.7
        {0.0, 1.0, -0.8}   // y - 0.8 = 0 => y = 0.8
    };
    double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 2);
    EXPECT_NEAR(0.56, area, EPSILON);  // 0.7 * 0.8 = 0.56
}

// Test with non-intersecting planes (resulting in empty area)
TEST(PolyVolFracTest, EmptyIntersection) {
    double planes[2][3] = {
        {1.0, 0.0, -0.5},   // x = 0.5 (right half)
        {-1.0, 0.0, 0.6}   // x = 0.4 (left half)
    };
    double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 2);
    EXPECT_NEAR(0.0, area, EPSILON);
}

// Test with triangle formation
TEST(PolyVolFracTest, TriangleCut) {
    // Diagonal: y = x
    // Vertical: x = 0.6
    double planes[2][3] = {
        {-1.0, 1.0, 0.0},   // -x + y = 0 => y = x
        {1.0, 0.0, -0.6}    // -x = -0.6 => x = 0.6
    };
    double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 2);
    // Triangle area = 0.5 * base * height = 0.5 * 0.6 * 0.6 = 0.18
    EXPECT_NEAR(0.18, area, EPSILON);
}

// Test with non-unit box
TEST(PolyVolFracTest, NonUnitBox) {
    // Box from (-2,-1) to (3,4)
    double x_min = -2.0, x_max = 3.0;
    double y_min = -1.0, y_max = 4.0;
    
    // Horizontal cut at y = 2.0
    double planes[1][3] = {
        {0.0, 1.0, -2.0}  // y = 2.0
    };
    double area = PlaneBoxIntersection(x_min, x_max, y_min, y_max, planes, 1);
    // Original box: 5 × 5 = 25
    // After cut: 5 × 3 = 15 (from y=-1 to y=2)
    EXPECT_NEAR(15.0, area, EPSILON);
}

// Test with multiple complex cuts
TEST(PolyVolFracTest, ComplexCuts) {
    // Three cuts to form a pentagon
    double planes[3][3] = {
        {1.0, 0.0, -0.8},    // x = 0.8 (vertical)
        {0.0, 1.0, -0.7},    // y = 0.7 (horizontal) 
        {-1.0, -1.0, 0.2}    // x + y = 0.2 (diagonal)
    };
    double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 3);
    // Area will be smaller than 0.8*0.7 but greater than 0
    EXPECT_GT(area, 0.0);
    EXPECT_LT(area, 0.56);
}

// Test with a tiny sliver
TEST(PolyVolFracTest, TinySliver) {
    // Create a very small sliver in the corner
    double planes[2][3] = {
        {1.0, 0.0, -0.99},   // x = 0.99
        {0.0, 1.0, -0.98}    // y = 0.98
    };
    double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 2);
    // Expected area: 0.01 * 0.02 = 0.0002
    EXPECT_NEAR(0.9702, area, EPSILON);
}

// Test with planes exactly through corners
TEST(PolyVolFracTest, CornerCases) {
    // Two diagonals through the box
    double planes[2][3] = {
        {1.0, 1.0, -1.0},    // x + y = 1 (diagonal through (0,1) and (1,0))
        {1.0, -1.0, 0.0}     // x - y = 0 (diagonal through (0,0) and (1,1))
    };
    double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 2);
    // These should create a perfect quarter of the box
    EXPECT_NEAR(0.25, area, EPSILON);
}