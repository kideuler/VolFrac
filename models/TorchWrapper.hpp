#ifndef TORCHWRAPPER_HPP
#define TORCHWRAPPER_HPP

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>

class TorchWrapper {
    public:
        TorchWrapper(std::string model_path, std::string norm_path); 
        double Predict(double x, double y, double nx, double ny, double K);

    private:
        std::vector<torch::jit::IValue> inputs;
        torch::jit::script::Module module;
        torch::Tensor x_mean, x_std;
        double y_mean, y_std;
};

#endif