#include "gtest/gtest.h"
#include "../models/MatVec.hpp"
#include <cmath>

// Matrix Class Tests
TEST(MatrixTest, Construction) {
    // Test default constructor
    Matrix m1;
    EXPECT_EQ(m1.getRows(), 0);
    EXPECT_EQ(m1.getCols(), 0);
    
    // Test size constructor
    Matrix m2(3, 4);
    EXPECT_EQ(m2.getRows(), 3);
    EXPECT_EQ(m2.getCols(), 4);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            EXPECT_DOUBLE_EQ(m2(i, j), 0.0);
        }
    }
    
    // Test size constructor with default value
    Matrix m3(2, 3, 5.0);
    EXPECT_EQ(m3.getRows(), 2);
    EXPECT_EQ(m3.getCols(), 3);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(m3(i, j), 5.0);
        }
    }
    
    // Test copy constructor
    Matrix m4 = m3;
    EXPECT_EQ(m4.getRows(), 2);
    EXPECT_EQ(m4.getCols(), 3);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(m4(i, j), 5.0);
        }
    }
    
    // Test assignment
    m2 = m3;
    EXPECT_EQ(m2.getRows(), 2);
    EXPECT_EQ(m2.getCols(), 3);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(m2(i, j), 5.0);
        }
    }
}

TEST(MatrixTest, ElementAccess) {
    Matrix m(3, 4);
    
    // Test setting elements
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            m(i, j) = i * 4 + j;
        }
    }
    
    // Test getting elements
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            EXPECT_DOUBLE_EQ(m(i, j), i * 4 + j);
        }
    }
    
    // Test out-of-bounds access
    EXPECT_THROW(m(3, 0), std::out_of_range);
    EXPECT_THROW(m(0, 4), std::out_of_range);
}

TEST(MatrixTest, Addition) {
    Matrix m1(2, 2);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;
    
    Matrix m2(2, 2);
    m2(0, 0) = 5.0; m2(0, 1) = 6.0;
    m2(1, 0) = 7.0; m2(1, 1) = 8.0;
    
    Matrix result = m1 + m2;
    EXPECT_DOUBLE_EQ(result(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 12.0);
    
    // Test in-place addition
    m1 += m2;
    EXPECT_DOUBLE_EQ(m1(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(m1(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(m1(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(m1(1, 1), 12.0);
    
    // Test dimension mismatch
    Matrix m3(3, 2);
    EXPECT_THROW(m1 + m3, std::invalid_argument);
    EXPECT_THROW(m1 += m3, std::invalid_argument);
}

TEST(MatrixTest, Subtraction) {
    Matrix m1(2, 2);
    m1(0, 0) = 5.0; m1(0, 1) = 6.0;
    m1(1, 0) = 7.0; m1(1, 1) = 8.0;
    
    Matrix m2(2, 2);
    m2(0, 0) = 1.0; m2(0, 1) = 2.0;
    m2(1, 0) = 3.0; m2(1, 1) = 4.0;
    
    Matrix result = m1 - m2;
    EXPECT_DOUBLE_EQ(result(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 4.0);
    
    // Test in-place subtraction
    m1 -= m2;
    EXPECT_DOUBLE_EQ(m1(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(m1(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(m1(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(m1(1, 1), 4.0);
    
    // Test dimension mismatch
    Matrix m3(3, 2);
    EXPECT_THROW(m1 - m3, std::invalid_argument);
    EXPECT_THROW(m1 -= m3, std::invalid_argument);
}

TEST(MatrixTest, MatrixMultiplication) {
    Matrix m1(2, 3);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0; m1(0, 2) = 3.0;
    m1(1, 0) = 4.0; m1(1, 1) = 5.0; m1(1, 2) = 6.0;
    
    Matrix m2(3, 2);
    m2(0, 0) = 7.0; m2(0, 1) = 8.0;
    m2(1, 0) = 9.0; m2(1, 1) = 10.0;
    m2(2, 0) = 11.0; m2(2, 1) = 12.0;
    
    Matrix result = m1 * m2;
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    
    EXPECT_DOUBLE_EQ(result(0, 0), 58.0);  // 1*7 + 2*9 + 3*11
    EXPECT_DOUBLE_EQ(result(0, 1), 64.0);  // 1*8 + 2*10 + 3*12
    EXPECT_DOUBLE_EQ(result(1, 0), 139.0); // 4*7 + 5*9 + 6*11
    EXPECT_DOUBLE_EQ(result(1, 1), 154.0); // 4*8 + 5*10 + 6*12
    
    // Test dimension mismatch
    Matrix m3(2, 2);
    EXPECT_THROW(m1 * m3, std::invalid_argument);
}

TEST(MatrixTest, ScalarMultiplication) {
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;
    
    Matrix result = m * 2.0;
    EXPECT_DOUBLE_EQ(result(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 8.0);
    
    // Test in-place scalar multiplication
    m *= 2.0;
    EXPECT_DOUBLE_EQ(m(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 8.0);
    
    // Test non-member scalar multiplication (scalar * matrix)
    Matrix m2(2, 2, 1.0);
    Matrix result2 = 3.0 * m2;
    EXPECT_DOUBLE_EQ(result2(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(result2(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(result2(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(result2(1, 1), 3.0);
}

TEST(MatrixTest, Transpose) {
    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;
    
    Matrix result = m.transpose();
    EXPECT_EQ(result.getRows(), 3);
    EXPECT_EQ(result.getCols(), 2);
    
    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(result(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(result(2, 1), 6.0);
}

// Vector Class Tests
TEST(VectorTest, Construction) {
    // Test default constructor
    Vector v1;
    EXPECT_EQ(v1.getSize(), 0);
    
    // Test size constructor
    Vector v2(4);
    EXPECT_EQ(v2.getSize(), 4);
    for (size_t i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(v2(i), 0.0);
    }
    
    // Test size constructor with default value
    Vector v3(3, 2.5);
    EXPECT_EQ(v3.getSize(), 3);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(v3(i), 2.5);
    }
    
    // Test copy constructor
    Vector v4 = v3;
    EXPECT_EQ(v4.getSize(), 3);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(v4(i), 2.5);
    }
    
    // Test assignment
    v2 = v3;
    EXPECT_EQ(v2.getSize(), 3);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(v2(i), 2.5);
    }
}

TEST(VectorTest, ElementAccess) {
    Vector v(4);
    
    // Test setting elements with () operator
    for (size_t i = 0; i < 4; i++) {
        v(i) = i + 1.0;
    }
    
    // Test getting elements with () operator
    for (size_t i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(v(i), i + 1.0);
    }
    
    // Test setting elements with [] operator
    for (size_t i = 0; i < 4; i++) {
        v[i] = 2.0 * (i + 1.0);
    }
    
    // Test getting elements with [] operator
    for (size_t i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(v[i], 2.0 * (i + 1.0));
    }
    
    // Test out-of-bounds access with () operator
    EXPECT_THROW(v(4), std::out_of_range);
}

TEST(VectorTest, Addition) {
    Vector v1(3);
    v1(0) = 1.0; v1(1) = 2.0; v1(2) = 3.0;
    
    Vector v2(3);
    v2(0) = 4.0; v2(1) = 5.0; v2(2) = 6.0;
    
    Vector result = v1 + v2;
    EXPECT_DOUBLE_EQ(result(0), 5.0);
    EXPECT_DOUBLE_EQ(result(1), 7.0);
    EXPECT_DOUBLE_EQ(result(2), 9.0);
    
    // Test in-place addition
    v1 += v2;
    EXPECT_DOUBLE_EQ(v1(0), 5.0);
    EXPECT_DOUBLE_EQ(v1(1), 7.0);
    EXPECT_DOUBLE_EQ(v1(2), 9.0);
    
    // Test dimension mismatch
    Vector v3(4);
    EXPECT_THROW(v1 + v3, std::invalid_argument);
    EXPECT_THROW(v1 += v3, std::invalid_argument);
}

TEST(VectorTest, Subtraction) {
    Vector v1(3);
    v1(0) = 4.0; v1(1) = 5.0; v1(2) = 6.0;
    
    Vector v2(3);
    v2(0) = 1.0; v2(1) = 2.0; v2(2) = 3.0;
    
    Vector result = v1 - v2;
    EXPECT_DOUBLE_EQ(result(0), 3.0);
    EXPECT_DOUBLE_EQ(result(1), 3.0);
    EXPECT_DOUBLE_EQ(result(2), 3.0);
    
    // Test in-place subtraction
    v1 -= v2;
    EXPECT_DOUBLE_EQ(v1(0), 3.0);
    EXPECT_DOUBLE_EQ(v1(1), 3.0);
    EXPECT_DOUBLE_EQ(v1(2), 3.0);
    
    // Test dimension mismatch
    Vector v3(4);
    EXPECT_THROW(v1 - v3, std::invalid_argument);
    EXPECT_THROW(v1 -= v3, std::invalid_argument);
}

TEST(VectorTest, ScalarMultiplication) {
    Vector v(3);
    v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
    
    Vector result = v * 2.0;
    EXPECT_DOUBLE_EQ(result(0), 2.0);
    EXPECT_DOUBLE_EQ(result(1), 4.0);
    EXPECT_DOUBLE_EQ(result(2), 6.0);
    
    // Test in-place scalar multiplication
    v *= 2.0;
    EXPECT_DOUBLE_EQ(v(0), 2.0);
    EXPECT_DOUBLE_EQ(v(1), 4.0);
    EXPECT_DOUBLE_EQ(v(2), 6.0);
    
    // Test non-member scalar multiplication (scalar * vector)
    Vector v2(3, 1.0);
    Vector result2 = 3.0 * v2;
    EXPECT_DOUBLE_EQ(result2(0), 3.0);
    EXPECT_DOUBLE_EQ(result2(1), 3.0);
    EXPECT_DOUBLE_EQ(result2(2), 3.0);
}

TEST(VectorTest, DotProduct) {
    Vector v1(3);
    v1(0) = 1.0; v1(1) = 2.0; v1(2) = 3.0;
    
    Vector v2(3);
    v2(0) = 4.0; v2(1) = 5.0; v2(2) = 6.0;
    
    double result = v1.dot(v2);
    EXPECT_DOUBLE_EQ(result, 1.0*4.0 + 2.0*5.0 + 3.0*6.0); // 32.0
    
    // Test dimension mismatch
    Vector v3(4);
    EXPECT_THROW(v1.dot(v3), std::invalid_argument);
}

TEST(VectorTest, Norm) {
    Vector v(3);
    v(0) = 3.0; v(1) = 4.0; v(2) = 0.0;
    
    double norm = v.norm();
    EXPECT_DOUBLE_EQ(norm, 5.0); // sqrt(3^2 + 4^2)
}

TEST(VectorTest, Normalize) {
    Vector v(3);
    v(0) = 3.0; v(1) = 4.0; v(2) = 0.0;
    
    Vector result = v.normalized();
    EXPECT_DOUBLE_EQ(result(0), 0.6);  // 3/5
    EXPECT_DOUBLE_EQ(result(1), 0.8);  // 4/5
    EXPECT_DOUBLE_EQ(result(2), 0.0);
    
    // Test in-place normalization
    v.normalize();
    EXPECT_DOUBLE_EQ(v(0), 0.6);
    EXPECT_DOUBLE_EQ(v(1), 0.8);
    EXPECT_DOUBLE_EQ(v(2), 0.0);
    EXPECT_DOUBLE_EQ(v.norm(), 1.0);
    
    // Test zero vector normalization (should not change and not throw)
    Vector zero(3, 0.0);
    zero.normalize();  // Should not throw
    EXPECT_DOUBLE_EQ(zero(0), 0.0);
    EXPECT_DOUBLE_EQ(zero(1), 0.0);
    EXPECT_DOUBLE_EQ(zero(2), 0.0);
}

// Matrix-Vector Multiplication Test
TEST(MatrixVectorTest, Multiplication) {
    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;
    
    Vector v(3);
    v(0) = 7.0; v(1) = 8.0; v(2) = 9.0;
    
    Vector result = m * v;
    EXPECT_EQ(result.getSize(), 2);
    EXPECT_DOUBLE_EQ(result(0), 1.0*7.0 + 2.0*8.0 + 3.0*9.0); // 50
    EXPECT_DOUBLE_EQ(result(1), 4.0*7.0 + 5.0*8.0 + 6.0*9.0); // 122
    
    // Test dimension mismatch
    Vector v2(2);
    EXPECT_THROW(m * v2, std::invalid_argument);
}

// Special edge cases and larger matrices/vectors
TEST(MatrixVectorAdvancedTest, EdgeCases) {
    // Create a larger matrix and test operations
    Matrix large(10, 10, 1.0);
    Matrix identity(10, 10, 0.0);
    for (size_t i = 0; i < 10; i++) {
        identity(i, i) = 1.0;
    }
    
    // Check that A*I = A
    Matrix result = large * identity;
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            EXPECT_DOUBLE_EQ(result(i, j), large(i, j));
        }
    }
    
    // Create a vector of ones
    Vector ones(10, 1.0);
    
    // Check that matrix * vector of ones produces row sums
    Vector rowSums = large * ones;
    for (size_t i = 0; i < 10; i++) {
        EXPECT_DOUBLE_EQ(rowSums(i), 10.0); // Sum of row is 10 (10 ones)
    }
}
