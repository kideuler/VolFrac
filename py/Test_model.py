import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# import the moel from Train.py
from Model import NeuralNet, NormalizationModule

model = NeuralNet(5,1)
PATH = 'models/best_model.pth'
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()

# load the normalization parameters
norm_path = 'models/normalization.pt'
norm_module = torch.jit.load(norm_path)

print(norm_module)

for i in range(500):
    x = random.random()
    y = random.random()
    nx = 2*random.random()-1
    ny = 2*random.random()-1
    norm = np.sqrt(nx*nx + ny*ny)
    nx /= norm
    ny /= norm
    k = np.log10(random.random()+1e-6)

    input = torch.tensor([x, y, nx, ny, k], dtype=torch.float32)
    input = (input - norm_module.x_mean) / norm_module.x_std

    # Get model prediction
    with torch.no_grad():
        output_normalized = model(input.unsqueeze(0)).squeeze()
        output = output_normalized * norm_module.y_std + norm_module.y_mean

    print(output.item())

# print normalization values
print(norm_module.x_mean)
print(norm_module.x_std)
print(norm_module.y_mean)
print(norm_module.y_std)