import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import getopt, sys
import os

from Model import *

### Beginning of Script

# Parse command line arguments
argumentList = sys.argv[1:]
options = "hd:s:m:n:"
long_options = ["help","data-path=", "save-path=","model-path=","num-epochs="]

# checking each argument
plot = True
path = 'build/data/VolFracData.dat'
save_path = 'models/'
load_model = False
model_path = 'models/model_weights.pth'
save_frequency = 0
num_epochs = 500

try:
    arguments, values = getopt.getopt(argumentList, options, long_options)
    
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print ("displaying help")
            print ("python Train.py")
            print ("-d --data-path: path to the data file")
            print ("-s --save-path: path to save the model")
            print ("-m --model-path: path to load the model")
            print ("-n --num-epochs: number of epochs to train")
            sys.exit(0)
        if currentArgument in ("-d", "--data-path"):
            path = currentValue
        if currentArgument in ("-s", "--save-path"):
            save_path = currentValue
        if currentArgument in ("-m", "--model-path"):
            model_path = currentValue
            load_model = True
        elif currentArgument in ("-n", "--num-epochs"):
            num_epochs = int(currentValue)

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))

dataset = VolFracDataSet(path)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
n_samples, n_features = dataset.x_train.shape


model = NeuralNet(input_size=n_features, num_classes=1)
learning_rate = 0.0001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

if load_model:
    model = torch.load(model_path, weights_only=False)
    model.eval()

norm_module = NormalizationModule(
    dataset.x_mean,
    dataset.x_std,
    dataset.y_mean.item(),
    dataset.y_std.item()
)
torch.jit.script(norm_module).save(save_path + "/normalization.pt")

# Training loop with early stopping
best_val_loss = float('inf')
patience = 20
patience_counter = 0
grad_clip = 1.0

# Use before training:
best_lr = find_lr(model, criterion, optimizer, dataloader)
# learning_rate = best_lr
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

warmup_epochs = 5
for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * (epoch + 1) / warmup_epochs
    model.train()
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_pred = model(dataset.x_val)
        val_loss = criterion(val_pred, dataset.y_val)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            example = torch.rand(1, 5)
            traced_script_module = torch.jit.trace(model, example)
            traced_script_module.save(save_path+"/VolFrac.pt")
            torch.save(model.state_dict(), save_path + "/best_model.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
            
        print(f'Epoch: {epoch+1}, Train Loss: {loss:.16f}, Val Loss: {val_loss:.16f}')

# Write the model data to a file

# write norm_module to a file
with open(save_path + "/model.dat", "w") as f:

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
        
    