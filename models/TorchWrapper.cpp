#include "TorchWrapper.hpp"

#ifdef USE_TORCH
TorchWrapper::TorchWrapper(std::string model_path, std::string norm_path) {
    try {
        module = torch::jit::load(model_path);
        
        // // Load normalization parameters
        // torch::jit::script::Module norm_data = torch::jit::load(norm_path);
        // x_mean = norm_data.attr("x_mean").toTensor();
        // x_std = norm_data.attr("x_std").toTensor();
        // y_mean = norm_data.attr("y_mean").toTensor()[0].item<double>();
        // y_std = norm_data.attr("y_std").toTensor()[0].item<double>();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model or normalization parameters\n";
    }
    
    inputs.push_back(torch::ones({1, 5}));
}

double TorchWrapper::Predict(double x, double y, double nx, double ny, double K) {
    // Apply normalization
    // inputs[0].toTensor()[0][0] = (x - x_mean[0].item<float>()) / x_std[0].item<float>();
    // inputs[0].toTensor()[0][1] = (y - x_mean[1].item<float>()) / x_std[1].item<float>();
    // inputs[0].toTensor()[0][2] = (nx - x_mean[2].item<float>()) / x_std[2].item<float>();
    // inputs[0].toTensor()[0][3] = (ny - x_mean[3].item<float>()) / x_std[3].item<float>();
    // inputs[0].toTensor()[0][4] = (K - x_mean[4].item<float>()) / x_std[4].item<float>();

    inputs[0].toTensor()[0][0] = x;
    inputs[0].toTensor()[0][1] = y;
    inputs[0].toTensor()[0][2] = nx;
    inputs[0].toTensor()[0][3] = ny;
    inputs[0].toTensor()[0][4] = K;
    
    // Get prediction and denormalize
    at::Tensor output = module.forward(inputs).toTuple()->elements()[0].toTensor();
    return output[0][0].item<double>();
}

#endif