import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import getopt, sys
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import datetime

class DualScheduler:
    def __init__(self, optimizer, min_lr=1e-6, max_lr=0.1, patience=20, factor=0.5, 
                 shock_threshold=50, shock_factor=2.0, max_shock_lr=0.1):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        self.shock_threshold = shock_threshold  # How many epochs of no improvement before shock
        self.shock_factor = shock_factor  # How much to increase LR during shock
        self.max_shock_lr = max_shock_lr  # Maximum LR during shock
        
        self.best_loss = float('inf')
        self.stagnation_counter = 0
        self.plateau_counter = 0
        self.mode = 'refine'  # Start in refinement mode
        
    def step(self, val_loss):
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Check if we improved
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.stagnation_counter = 0
            self.plateau_counter = 0
            return current_lr  # No change needed
        else:
            self.stagnation_counter += 1
            
            # In refinement mode, we reduce learning rate after patience steps
            if self.mode == 'refine':
                self.plateau_counter += 1
                
                # If we've been on a plateau for 'patience' steps, reduce LR
                if self.plateau_counter >= self.patience:
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    if new_lr != current_lr:
                        print(f"Reducing LR from {current_lr:.6f} to {new_lr:.6f}")
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        self.plateau_counter = 0
                        
                # If we've been stagnant for shock_threshold steps, switch to shock mode
                if self.stagnation_counter >= self.shock_threshold:
                    self.mode = 'shock'
                    new_lr = min(current_lr * self.shock_factor, self.max_shock_lr)
                    print(f"Shocking system! Increasing LR from {current_lr:.6f} to {new_lr:.6f}")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.plateau_counter = 0
            
            # In shock mode, we switch back to refinement after one step
            elif self.mode == 'shock':
                self.mode = 'refine'
                self.plateau_counter = 0
                self.stagnation_counter = 0
                self.shock_threshold *= 2  # Double the shock threshold
        
        return self.optimizer.param_groups[0]['lr']

class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error Loss:
    sum(|pred - target|) / sum(|target|)
    
    This measures the total absolute error as a percentage of the total target value.
    """
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # Add small epsilon to prevent division by zero
        absolute_errors = torch.abs(pred - target)
        sum_absolute_errors = torch.sum(absolute_errors)
        sum_absolute_targets = torch.sum(torch.abs(target)) + self.epsilon
        
        return sum_absolute_errors / sum_absolute_targets

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
    
class VolFracDataSet_Plane(Dataset):
    def __init__(self, filename):
        xy = np.loadtxt(filename, delimiter=',', dtype=np.float32)
        self.x = torch.from_numpy(xy[:, [0,1,2,3]].astype(np.float32))
        self.y = torch.from_numpy(xy[:, 4].astype(np.float32)).view(-1, 1)
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
            # He initialization (better for ReLU activation)
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
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

# Replace with your actual token and channel ID
slack_token = os.environ.get("SLACK_BOT_TOKEN")  # It is recommended to store the token as an environment variable
channel_id = "C08KL7ASXEU"  # or 'your_user_id' if you want to send a direct message

# Initialize the Slack client
client = WebClient(token=slack_token)

def send_message_with_file_to_slack(message, file_path, file_title=None, file_comment=None):
    """
    Send a message and a file to a Slack channel
    
    Args:
        message: Text message to send
        file_path: Path to the file to upload
        file_title: Optional title for the file (defaults to filename)
        file_comment: Optional comment for the file
    """
    try:
        # Send the text message first
        message_response = client.chat_postMessage(
            channel=channel_id,
            text=message
        )
        print(f"Message sent: {message_response['ts']}")
        
        # Extract filename from path if no title provided
        if file_title is None:
            file_title = os.path.basename(file_path)
            
        # Upload the file
        file_response = client.files_upload_v2(
            channel=channel_id,
            file=file_path,
            title=file_title,
            initial_comment=file_comment)
        print(f"File uploaded: {file_response['file']['id']}")
        
        return True
        
    except SlackApiError as e:
        print(f"Error: {e}")
        return False