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
        xy[:, 4] = np.log10(xy[:, 4] + 1e-10) # Log transform the curvature
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
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        
        # Define network architecture with 4 hidden layers (matching the original's depth)
        # Layer sizes match the original's progression: 64 -> 128 -> 128 -> 64
        self.layer1 = nn.Linear(input_size, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, num_classes)
        
        # Activation function
        self.activation = nn.ReLU()  # SiLU/Swish often performs better than ReLU for regression
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
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