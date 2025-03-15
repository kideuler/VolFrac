import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import getopt, sys
import os

def find_lr(model, criterion, optimizer, dataloader, start_lr=1e-7, end_lr=1, num_iter=100):
    # Save original model parameters
    original_params = [p.clone().detach() for p in model.parameters()]
    
    lrs = []
    losses = []
    best_loss = float('inf')
    
    # Update learning rate logarithmically
    mult = (end_lr/start_lr)**(1/num_iter)
    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr
    
    model.train()
    for i, (x_batch, y_batch) in enumerate(dataloader):
        if i >= num_iter:
            break
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        # Store values
        lrs.append(lr)
        if loss < best_loss:
            best_loss = loss.item()
        
        # Stop if loss explodes
        if loss > 4 * best_loss or torch.isnan(loss):
            break
            
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    
    # Restore original parameters
    for p, p_orig in zip(model.parameters(), original_params):
        p.data.copy_(p_orig.data)
    
    # Find best learning rate (usually order of magnitude less than minimum)
    min_idx = np.argmin(losses)
    best_lr = lrs[min_idx] / 10
    print(f"Suggested learning rate: {best_lr:.8f}")
    return best_lr

class RelativeMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(RelativeMSELoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # Add small epsilon to prevent division by zero
        relative_error = (pred - target) / (target.abs() + self.epsilon)
        return torch.mean(relative_error ** 2)

class VolFracDataSet(Dataset):
    def __init__(self, filename):
        xy = np.loadtxt(filename, delimiter=',', dtype=np.float32)
        # xy[:, 4] = np.log10(xy[:, 4] + 1e-10) # Log transform the curvature
        self.x = torch.from_numpy(xy[:, [0,1,2,3,4]].astype(np.float32))
        self.y = torch.from_numpy(xy[:, 5].astype(np.float32)).view(-1, 1)
        print(self.x.min())
        print(self.x.max())

        # Split into train/val/test
        x_temp, self.x_test, y_temp, self.y_test = train_test_split(self.x, self.y, test_size=0.2)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_temp, y_temp, test_size=0.2)
        
        # Normalize inputs
        self.x_mean = self.x.mean(dim=0)
        self.x_std = self.x.std(dim=0)
        self.x_train = (self.x_train - self.x_mean) / self.x_std
        self.x_val = (self.x_val - self.x_mean) / self.x_std
        self.x_test = (self.x_test - self.x_mean) / self.x_std
        
        # Normalize targets
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()
        self.y_train = (self.y_train - self.y_mean) / self.y_std
        self.y_val = (self.y_val - self.y_mean) / self.y_std
        self.y_test = (self.y_test - self.y_mean) / self.y_std
        
        self.n_samples = self.x_train.shape[0]

        # print the mean and std of the training data
        
        print(self.x_mean)
        print(self.x_std)
        print(self.y.min())
        print(self.y.max())
        print(self.y_mean)
        print(self.y_std)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    
    def get_test_data(self):
        return self.x_test, self.y_test

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 64, 128, 64, 32], num_classes=1):
        super(NeuralNet, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
        
        # Final output layer (no activation)
        x = self.output_layer(x)
        return x

# Normalization Module
class NormalizationModule(torch.nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_std', x_std)
        self.register_buffer('y_mean', torch.tensor([y_mean]))
        self.register_buffer('y_std', torch.tensor([y_std]))


def Write_model(filename,norm_module,dataset,model):
    with open(filename, "w") as f:

        f.write(f"# data means\n")
        for x in norm_module.x_mean.tolist():
            f.write(f"{x} ")
        f.write(f"{dataset.y_mean}\n")
        f.write(f"\n# data standard deviation\n")
        for x in norm_module.x_std.tolist():
            f.write(f"{x} ")
        f.write(f"{dataset.y_std}\n")

        # Write model architecture information
        f.write(f"\n# Model Architecture\n")
        
        # Get activation function name
        activation_name = model.activation.__class__.__name__
        f.write(f"Activation_function = {activation_name}\n")
        
        # Number of layers
        num_layers = len([name for name, _ in model.named_parameters() if 'weight' in name])
        f.write(f"Number_of_layers = {num_layers}\n")

        # Write parameters for each layer
        for name, param in model.named_parameters():
            f.write(f"\n# Layer: {name}\n")
            
            # For weights (2D tensors)
            if len(param.shape) == 2:
                sz = list(param.shape)
                f.write(f"{sz[0]} {sz[1]}\n")
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        f.write(f"{param[i,j].item():.12f} ")
                    f.write(f"\n")
            # For biases (1D tensors)
            elif len(param.shape) == 1:
                sz = list(param.shape)
                f.write(f"{sz[0]}\n")
                for i in range(param.shape[0]):
                    f.write(f"{param[i].item():.12f} ")
                f.write(f"\n")
            else:
                f.write(f"# Skipping parameter with unusual shape: {param.shape}\n")