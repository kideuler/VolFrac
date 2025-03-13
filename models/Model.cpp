#include "Model.hpp"

Model::Model(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model file: " + filename);
    }

    // Read means and standard deviations
    std::string line;
    
    // Skip "# data means" line
    std::getline(file, line);
    
    // Parse means
    std::getline(file, line);
    std::istringstream means_stream(line);
    std::vector<double> means;
    double value;
    while (means_stream >> value) {
        means.push_back(value);
    }
    
    // Skip empty line and "# data standard deviation" line
    std::getline(file, line);
    std::getline(file, line);
    
    // Parse standard deviations
    std::getline(file, line);
    std::istringstream stds_stream(line);
    std::vector<double> stds;
    while (stds_stream >> value) {
        stds.push_back(value);
    }
    
    // Input/Output normalization
    input_mean = Vector(5);
    input_std = Vector(5);
    for (int i = 0; i < 5; ++i) {
        input_mean(i) = means[i];
        input_std(i) = stds[i];
    }
    output_mean = means[5];
    output_std = stds[5];
    
    // Skip lines until we reach "# Model Architecture"
    while (std::getline(file, line) && line != "# Model Architecture");
    
    // Read activation function
    std::getline(file, line);
    if (line.find("ReLU") != std::string::npos) {
        activation_function = "ReLU";
    } else if (line.find("Tanh") != std::string::npos) {
        activation_function = "Tanh";
    } else if (line.find("SiLU") != std::string::npos) {
        activation_function = "SiLU";
    } else {
        activation_function = "Unknown";
    }

    // Set activation function based on string from file
    auto it = activationMap.find(activation_function);
    if (it != activationMap.end()) {
        // Found the activation function in the map
        activation = it->second;
    } else {
        // Default to identity function if not found
        activation = [](double x) { return x; };
        std::cerr << "Warning: Unknown activation function '" << activation_function 
                  << "'. Using identity function instead." << std::endl;
    }
    
    // Read number of layers
    std::getline(file, line);
    std::istringstream layers_stream(line);
    std::string key, equal_sign;
    int num_layers;
    layers_stream >> key >> equal_sign >> num_layers;
    
    // Initialize weights and biases vectors
    weights.clear();
    biases.clear();
    
    // Read each layer
    for (int layer = 0; layer < num_layers; ++layer) {
        // Skip empty line if any
        if (!std::getline(file, line) || line.empty()) {
            std::getline(file, line);
        }
        
        // Read matrix dimensions
        std::getline(file, line);
        std::istringstream dims_stream(line);
        int rows, cols;
        dims_stream >> rows >> cols;
        
        // Create weight matrix
        Matrix weight_matrix(rows, cols);
        
        // Read weight matrix values
        for (int i = 0; i < rows; ++i) {
            std::getline(file, line);
            std::istringstream row_stream(line);
            for (int j = 0; j < cols; ++j) {
                row_stream >> value;
                weight_matrix(i, j) = value;
            }
        }
        
        weights.push_back(weight_matrix);
        
        // Skip empty line if any
        if (!std::getline(file, line) || line.empty()) {
            std::getline(file, line);
        }
        
        
        // Read bias dimensions
        std::getline(file, line);
        std::istringstream bias_dims_stream(line);
        int bias_size;
        bias_dims_stream >> bias_size;
        
        // Create bias vector
        Vector bias_vector(bias_size);
        
        // Read bias values
        std::getline(file, line);
        std::istringstream bias_stream(line);
        for (int j = 0; j < bias_size; ++j) {
            bias_stream >> value;
            bias_vector(j) = value;
        }
        
        biases.push_back(bias_vector);
    }
    
    // Validate that we have the correct number of weights and biases
    if (weights.size() != biases.size()) {
        throw std::runtime_error("Mismatch in number of weight and bias layers");
    }
    
    std::cout << "Model loaded successfully from " << filename << std::endl;
    std::cout << "Number of layers: " << weights.size() << std::endl;
    std::cout << "Activation function: " << activation_function << std::endl;
}

void Model::PrintModel() {
    std::cout << "Input mean: " << input_mean << std::endl;
    std::cout << "Input std: " << input_std << std::endl;
    std::cout << "Output mean: " << output_mean << std::endl;
    std::cout << "Output std: " << output_std << std::endl;
    std::cout << "Activation function: " << activation_function << std::endl;
    
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << "Layer " << i << " weights:" << std::endl;
        Matrix temp = weights[i];
        for (size_t j = 0; j < temp.getRows(); ++j) {
            for (size_t k = 0; k < temp.getCols(); ++k) {
                std::cout << temp(j, k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Layer " << i << " biases:" << std::endl;
        Vector temp2 = biases[i];
        for (size_t j = 0; j < temp2.getSize(); ++j) {
            std::cout << temp2(j) << " ";
        }
        std::cout << std::endl;
    }
}

double Model::Predict(double input[5]) {
    // Normalize input
    Vector input_vec(5);
    for (int i = 0; i < 5; ++i) {
        input_vec(i) = input[i];
    }

    // fifth input is log10
    input_vec[4] = std::log10(input_vec[4]);
    for (int i = 0; i < 5; ++i) {
        input_vec(i) = (input_vec(i) - input_mean(i)) / input_std(i);
    }
    
    // Forward pass
    Vector output;
    for (size_t i = 0; i < weights.size(); ++i) {
        output = weights[i] * input_vec + biases[i];
        applyActivation(output);
        input_vec = output;
    }
    
    // Denormalize output
    double Vf = output(0) * output_std + output_mean;

    if (Vf < 0.0) {
        return 0.0;
    } else if (Vf > 1.0) {
        return 1.0;
    } else {
        return Vf;
    }
}