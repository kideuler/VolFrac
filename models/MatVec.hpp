#ifndef __MATVEC_HPP__
#define __MATVEC_HPP__

#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath> // For sqrt in norm calculations

/**
 * Vector class with manual memory management
 */
class Vector {
    private:
        double* data;     // Pointer to array storage
        size_t size;      // Number of elements
        
    public:
        // Default constructor
        Vector() : data(nullptr), size(0) {}
        
        // Constructor with dimension
        Vector(size_t n) : size(n) {
            if (size > 0) {
                data = new double[size]();  // Allocate and zero-initialize
            } else {
                data = nullptr;
                size = 0;
            }
        }
        
        // Constructor with dimension and default value
        Vector(size_t n, double defaultValue) : size(n) {
            if (size > 0) {
                data = new double[size];
                for (size_t i = 0; i < size; i++) {
                    data[i] = defaultValue;
                }
            } else {
                data = nullptr;
                size = 0;
            }
        }
        
        // Copy constructor
        Vector(const Vector& other) : size(other.size) {
            if (size > 0) {
                data = new double[size];
                std::memcpy(data, other.data, size * sizeof(double));
            } else {
                data = nullptr;
            }
        }
        
        // Move constructor
        Vector(Vector&& other) noexcept : data(other.data), size(other.size) {
            other.data = nullptr;
            other.size = 0;
        }
        
        // Destructor
        ~Vector() {
            delete[] data;  // Safe to call delete[] on nullptr
        }
        
        // Assignment operator
        Vector& operator=(const Vector& other) {
            if (this != &other) {  // Self-assignment check
                // Free old memory
                delete[] data;
                
                // Copy size
                size = other.size;
                
                // Allocate and copy data
                if (size > 0) {
                    data = new double[size];
                    std::memcpy(data, other.data, size * sizeof(double));
                } else {
                    data = nullptr;
                }
            }
            return *this;
        }
        
        // Move assignment operator
        Vector& operator=(Vector&& other) noexcept {
            if (this != &other) {
                delete[] data;
                
                // Take ownership of other's resources
                data = other.data;
                size = other.size;
                
                // Reset other
                other.data = nullptr;
                other.size = 0;
            }
            return *this;
        }
        
        // Element access operator with bounds checking
        double& operator()(size_t index) {
            if (index >= size) {
                throw std::out_of_range("Vector index out of range");
            }
            return data[index];
        }
        
        // Const element access operator
        const double& operator()(size_t index) const {
            if (index >= size) {
                throw std::out_of_range("Vector index out of range");
            }
            return data[index];
        }
        
        // Array-style access (no bounds checking for efficiency)
        double& operator[](size_t index) {
            return data[index];
        }
        
        const double& operator[](size_t index) const {
            return data[index];
        }
        
        // Get size
        size_t getSize() const { return size; }
        
        // Vector addition
        Vector operator+(const Vector& other) const {
            if (size != other.size) {
                throw std::invalid_argument("Vector dimensions must match for addition");
            }
            
            Vector result(size);
            for (size_t i = 0; i < size; i++) {
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }
        
        // In-place vector addition
        Vector& operator+=(const Vector& other) {
            if (size != other.size) {
                throw std::invalid_argument("Vector dimensions must match for addition");
            }
            
            for (size_t i = 0; i < size; i++) {
                data[i] += other.data[i];
            }
            return *this;
        }
        
        // Vector subtraction
        Vector operator-(const Vector& other) const {
            if (size != other.size) {
                throw std::invalid_argument("Vector dimensions must match for subtraction");
            }
            
            Vector result(size);
            for (size_t i = 0; i < size; i++) {
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }
        
        // In-place vector subtraction
        Vector& operator-=(const Vector& other) {
            if (size != other.size) {
                throw std::invalid_argument("Vector dimensions must match for subtraction");
            }
            
            for (size_t i = 0; i < size; i++) {
                data[i] -= other.data[i];
            }
            return *this;
        }
        
        // Scalar multiplication
        Vector operator*(double scalar) const {
            Vector result(size);
            for (size_t i = 0; i < size; i++) {
                result.data[i] = data[i] * scalar;
            }
            return result;
        }
        
        // In-place scalar multiplication
        Vector& operator*=(double scalar) {
            for (size_t i = 0; i < size; i++) {
                data[i] *= scalar;
            }
            return *this;
        }
        
        // Dot product
        double dot(const Vector& other) const {
            if (size != other.size) {
                throw std::invalid_argument("Vector dimensions must match for dot product");
            }
            
            double result = 0.0;
            for (size_t i = 0; i < size; i++) {
                result += data[i] * other.data[i];
            }
            return result;
        }
        
        // L2 Norm (Euclidean length)
        double norm() const {
            double sum = 0.0;
            for (size_t i = 0; i < size; i++) {
                sum += data[i] * data[i];
            }
            return std::sqrt(sum);
        }
        
        // Normalize vector (make unit length)
        Vector& normalize() {
            double n = norm();
            if (n > 0.0) {  // Prevent division by zero
                for (size_t i = 0; i < size; i++) {
                    data[i] /= n;
                }
            }
            return *this;
        }
        
        // Create a normalized copy
        Vector normalized() const {
            Vector result(*this);
            result.normalize();
            return result;
        }
        
        // Print vector contents
        friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
            os << "[ ";
            for (size_t i = 0; i < v.size; i++) {
                os << v.data[i];
                if (i < v.size - 1) os << ", ";
            }
            os << " ]";
            return os;
        }
};

// Non-member scalar multiplication (scalar * vector)
inline Vector operator*(double scalar, const Vector& vec) {
    return vec * scalar;
}

/**
 * Matrix class with manual memory management using 1D array storage
 * for better memory locality and performance
 */
class Matrix {
private:
    double* data;       // Pointer to flat 1D array storage
    size_t rows;        // Number of rows
    size_t cols;        // Number of columns

public:
    // Default constructor
    Matrix() : data(nullptr), rows(0), cols(0) {}
    
    // Constructor with dimensions
    Matrix(size_t numRows, size_t numCols) : rows(numRows), cols(numCols) {
        if (rows > 0 && cols > 0) {
            data = new double[rows * cols]();  // Allocate and zero-initialize
        } else {
            data = nullptr;
            rows = 0;
            cols = 0;
        }
    }
    
    // Constructor with dimensions and default value
    Matrix(size_t numRows, size_t numCols, double defaultValue) : rows(numRows), cols(numCols) {
        if (rows > 0 && cols > 0) {
            size_t size = rows * cols;
            data = new double[size];
            
            // Fill with default value
            for (size_t i = 0; i < size; i++) {
                data[i] = defaultValue;
            }
        } else {
            data = nullptr;
            rows = 0;
            cols = 0;
        }
    }
    
    // Copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        if (rows > 0 && cols > 0) {
            size_t size = rows * cols;
            data = new double[size];
            std::memcpy(data, other.data, size * sizeof(double));
        } else {
            data = nullptr;
        }
    }
    
    // Move constructor
    Matrix(Matrix&& other) noexcept : data(other.data), rows(other.rows), cols(other.cols) {
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    
    // Destructor
    ~Matrix() {
        delete[] data;  // Safe to call delete[] on nullptr
    }
    
    // Assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {  // Self-assignment check
            // Free old memory
            delete[] data;
            
            // Copy dimensions
            rows = other.rows;
            cols = other.cols;
            
            // Allocate and copy data
            if (rows > 0 && cols > 0) {
                size_t size = rows * cols;
                data = new double[size];
                std::memcpy(data, other.data, size * sizeof(double));
            } else {
                data = nullptr;
            }
        }
        return *this;
    }
    
    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] data;
            
            // Take ownership of other's resources
            data = other.data;
            rows = other.rows;
            cols = other.cols;
            
            // Reset other
            other.data = nullptr;
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }
    
    // Element access operator using i*cols+j indexing
    double& operator()(size_t row, size_t col) {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[row * cols + col];
    }
    
    // Const element access operator
    const double& operator()(size_t row, size_t col) const {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[row * cols + col];
    }
    
    // Get number of rows
    size_t getRows() const { return rows; }
    
    // Get number of columns
    size_t getCols() const { return cols; }
    
    // Addition operator
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        
        Matrix result(rows, cols);
        size_t size = rows * cols;
        
        for (size_t i = 0; i < size; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        
        return result;
    }
    
    // In-place addition operator
    Matrix& operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        
        size_t size = rows * cols;
        for (size_t i = 0; i < size; i++) {
            data[i] += other.data[i];
        }
        
        return *this;
    }
    
    // Subtraction operator
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        
        Matrix result(rows, cols);
        size_t size = rows * cols;
        
        for (size_t i = 0; i < size; i++) {
            result.data[i] = data[i] - other.data[i];
        }
        
        return result;
    }
    
    // In-place subtraction operator
    Matrix& operator-=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        
        size_t size = rows * cols;
        for (size_t i = 0; i < size; i++) {
            data[i] -= other.data[i];
        }
        
        return *this;
    }
    
    // Matrix multiplication operator
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }
        
        Matrix result(rows, other.cols);
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < cols; k++) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    }
    
    // Scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        size_t size = rows * cols;
        
        for (size_t i = 0; i < size; i++) {
            result.data[i] = data[i] * scalar;
        }
        
        return result;
    }
    
    // In-place scalar multiplication
    Matrix& operator*=(double scalar) {
        size_t size = rows * cols;
        
        for (size_t i = 0; i < size; i++) {
            data[i] *= scalar;
        }
        
        return *this;
    }
    
    // Matrix transposition
    Matrix transpose() const {
        Matrix result(cols, rows);
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        
        return result;
    }
    
    // Print matrix contents
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        for (size_t i = 0; i < m.rows; i++) {
            os << "[ ";
            for (size_t j = 0; j < m.cols; j++) {
                os << m(i, j);
                if (j < m.cols - 1) os << ", ";
            }
            os << " ]";
            if (i < m.rows - 1) os << std::endl;
        }
        return os;
    }

    Vector operator*(const Vector& vec) const {
        if (cols != vec.getSize()) {
            throw std::invalid_argument("Matrix and vector dimensions incompatible for multiplication");
        }
        
        Vector result(rows);
        
        for (size_t i = 0; i < rows; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < cols; j++) {
                sum += (*this)(i, j) * vec(j);
            }
            result(i) = sum;
        }
        
        return result;
    }
};

// Non-member scalar multiplication (scalar * matrix)
inline Matrix operator*(double scalar, const Matrix& mat) {
    return mat * scalar;  // Reuse the matrix * scalar implementation
}

#endif // __MATVEC_HPP__