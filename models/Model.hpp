#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include <sstream>
#include <cmath>
#include <unordered_map>

#include "MatVec.hpp"

/**
 * @file Model.hpp
 * @brief Neural network model implementation for volume fraction prediction
 *
 * This file provides a model class that loads and executes a trained neural network
 * for predicting volume fractions based on geometric features.
 */

/**
 * @brief ReLU activation function
 * 
 * Implements the Rectified Linear Unit activation function: f(x) = max(0, x)
 * 
 * @param x Input value
 * @return ReLU of input value (x if positive, 0 otherwise)
 */
inline double ReLU(double x) {
    return x > 0.0 ? x : 0.0;
}

/**
 * @brief Hyperbolic tangent activation function
 * 
 * Implements the tanh activation function: f(x) = tanh(x)
 * 
 * @param x Input value
 * @return Hyperbolic tangent of input value
 */
inline double Tanh(double x) {
    return std::tanh(x);
}

/**
 * @brief SiLU (Sigmoid Linear Unit) activation function
 * 
 * Implements the SiLU/Swish activation function: f(x) = x * sigmoid(x)
 * 
 * @param x Input value
 * @return SiLU of input value
 */
inline double SiLU(double x) {
    return x / (1.0 + std::exp(-x));
}

/**
 * @class Model
 * @brief Neural network model for inference
 * 
 * This class implements a fully connected neural network that can be loaded from
 * a file and used for making predictions. The model supports various activation
 * functions and includes input/output normalization.
 */
class Model {
    public:
        /**
         * @typedef ActivationFunction
         * @brief Function pointer type for activation functions
         */
        using ActivationFunction = std::function<double(double)>;

        /**
         * @brief Constructs a model from a saved file
         * 
         * Loads network architecture, weights, biases, and normalization parameters
         * from the specified file.
         * 
         * @param filename Path to the saved model file
         */
        Model(const std::string& filename);

        /**
         * @brief Prints model information to standard output
         * 
         * Displays the network architecture, including layer sizes, activation function,
         * and normalization parameters.
         */
        void PrintModel();

        /**
         * @brief Applies the activation function to each element in a vector
         * 
         * @param vec Vector to be transformed in-place
         */
        void applyActivation(Vector& vec) const {
            for (size_t i = 0; i < vec.getSize(); ++i) {
                vec(i) = activation(vec(i));
            }
        }

        /**
         * @brief Performs forward pass through the network to make a prediction
         * 
         * Takes input features, normalizes them, forwards them through the neural network,
         * and denormalizes the output.
         * 
         * @param input Array of input features
         * @return Predicted volume fraction value
         */
        double Predict(double input[]);

        /**
         * @brief Map of available activation functions by name
         * 
         * Associates string identifiers with activation function implementations
         * to support model loading with different activation functions.
         */
        const std::unordered_map<std::string, Model::ActivationFunction> activationMap = {
            {"ReLU", ReLU},
            {"Tanh", Tanh},
            {"SiLU", SiLU},
            // Add more activation functions as needed
        };

    private:
        std::vector<Matrix> weights;      ///< Weight matrices for each layer
        std::vector<Vector> biases;       ///< Bias vectors for each layer
        std::string activation_function;  ///< Name of the activation function
        ActivationFunction activation;    ///< Function pointer to the activation function
    
        // Normalization parameters
        Vector input_mean;    ///< Mean values for input feature normalization
        Vector input_std;     ///< Standard deviation values for input feature normalization
        double output_mean;   ///< Mean value for output denormalization
        double output_std;    ///< Standard deviation value for output denormalization

        int input_size;       ///< Number of input features expected by the model
};

#endif // __MODEL_HPP__