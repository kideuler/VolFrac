#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include <sstream>
#include <cmath>


#include "MatVec.hpp"

// Activation functions
inline double ReLU(double x) {
    return x > 0.0 ? x : 0.0;
}

inline double Tanh(double x) {
    return std::tanh(x);
}

inline double SiLU(double x) {
    return x / (1.0 + std::exp(-x));
}

class Model {
    public:
        // Activation function type
        using ActivationFunction = std::function<double(double)>;

        Model(const std::string& filename);

        void PrintModel();

        // Apply activation to a vector
        void applyActivation(Vector& vec) const {
            for (size_t i = 0; i < vec.getSize(); ++i) {
                vec(i) = activation(vec(i));
            }
        }

        double Predict(double input[5]);


    private:
        std::vector<Matrix> weights;
        std::vector<Vector> biases;
        std::string activation_function;
        ActivationFunction activation;
        static const std::unordered_map<std::string, ActivationFunction> activationMap;
    
        // Normalization parameters
        Vector input_mean;
        Vector input_std;
        double output_mean;
        double output_std;
};

// Define the static activation function map
const std::unordered_map<std::string, Model::ActivationFunction> Model::activationMap = {
    {"ReLU", ReLU},
    {"Tanh", Tanh},
    {"SiLU", SiLU},
    // Add more activation functions as needed
};

#endif // __MODEL_HPP__