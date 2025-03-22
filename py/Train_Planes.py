import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import getopt, sys
import os
from torch.optim import Adam, AdamW, RMSprop, Adagrad
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

from Model import *

### Beginning of Script

# Parse command line arguments
argumentList = sys.argv[1:]
options = "hd:s:m:n:P"
long_options = ["help","data-path=", "save-path=","model-path=","num-epochs=","plot"]

# checking each argument
plot = False
path = 'build/data/VolFracData_Plane.dat'
save_path = 'models/'
load_model = False
model_path = 'models/best_model_plane.pth'
save_frequency = 0
num_epochs = 1000
optimizer_name = "adamW"  # Options: "adam", "adamw", "rmsprop", "adagrad", "onecycle"
base_lr = 0.001

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
            print ("-P --plot: plot the results")
            sys.exit(0)
        if currentArgument in ("-d", "--data-path"):
            path = currentValue
        if currentArgument in ("-s", "--save-path"):
            save_path = currentValue
        if currentArgument in ("-m", "--model-path"):
            model_path = currentValue
            load_model = True
        if currentArgument in ("-P", "--plot"):
            plot = True
        elif currentArgument in ("-n", "--num-epochs"):
            num_epochs = int(currentValue)

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))

dataset = VolFracDataSet_Plane(path)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
n_samples, n_features = dataset.x_train.shape

model = NeuralNet(input_size=n_features, hidden_sizes=[64,64,64], num_classes=1)

norm_module = NormalizationModule(
    dataset.x_mean,
    dataset.x_std,
    dataset.y_mean.item(),
    dataset.y_std.item()
)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 100
patience_counter = 0
grad_clip = 1.0

# Choose optimizer based on option
if optimizer_name == "adam":
    optimizer = Adam(model.parameters(), lr=base_lr, weight_decay=0.01)
elif optimizer_name == "adamw":
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
elif optimizer_name == "rmsprop":
    optimizer = RMSprop(model.parameters(), lr=base_lr, weight_decay=1e-3, momentum=0.99)
elif optimizer_name == "adagrad":
    optimizer = Adagrad(model.parameters(), lr=base_lr, weight_decay=1e-5)
elif optimizer_name == "onecycle":
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    # OneCycleLR provides a learning rate that first increases then decreases
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=25,  # Initial LR will be max_lr/25
        final_div_factor=1000  # Final LR will be max_lr/1000
    )
else:
    # Default to AdamW
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

# Set up main scheduler - either we're using OneCycle (already configured above) 
# or we're using ReduceLROnPlateau for other optimizers
if optimizer_name != "onecycle":
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )


criterion = MAPELoss()

# Initialize lists to store loss history
train_losses = []
val_losses = []
lr_history = []

if load_model:
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

# Plotting
if plot:
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Set up the plots
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    # ax1.set_ylim(0, 1)
    train_line, = ax1.plot([], [], 'b-', label='Training Loss')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    # ax2.set_ylim(0, 1)
    val_line, = ax2.plot([], [], 'r-', label='Validation Loss')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show(block=False)

# Training loop
for epoch in range(num_epochs):

    model.train()
    running_train_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if optimizer_name != "onecycle":
            scheduler.step(epoch)

        running_train_loss += loss.item()
        num_batches += 1
    
    # Calculate average training loss
    avg_train_loss = running_train_loss / num_batches
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_pred = model(dataset.x_val)
        val_loss = criterion(val_pred, dataset.y_val)
        scheduler.step(val_loss)
        val_losses.append(val_loss.item())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            example = torch.rand(1, 4)
            traced_script_module = torch.jit.trace(model, example)
            traced_script_module.save(save_path+"/VolFrac.pt")
            torch.save(model.state_dict(), save_path + "/best_model.pth")
            Write_model(save_path + "/model_planes.dat", norm_module, dataset, model)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
            
        print(f'Epoch: {epoch+1}, Train Loss: {loss:.16f}, Val Loss: {val_loss:.16f}')
        
        # Update plots every 10 epochs or on the last epoch
        if plot and (epoch % 10 == 0 or epoch == num_epochs - 1 or patience_counter >= patience):
            epochs_range = list(range(1, len(train_losses) + 1))
            
            # Clear previous plot
            ax1.cla()
            ax2.cla()
            
            # Plot training loss
            ax1.plot(epochs_range, train_losses, 'b-')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            # ax1.set_ylim(0, 2)
            ax1.set_yscale('log')  # Use log scale for better visualization
            ax1.grid(True)
            
            # Plot validation loss
            ax2.plot(epochs_range, val_losses, 'r-')
            ax2.set_title('Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            # ax2.set_ylim(0, 2)
            ax2.set_yscale('log')  # Use log scale for better visualization
            ax2.grid(True)
            
            # Add best validation loss marker
            best_epoch = val_losses.index(min(val_losses)) + 1
            ax2.plot(best_epoch, min(val_losses), 'go', markersize=8, label=f'Best: {min(val_losses):.6f}')
            ax2.legend()
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)  # Pause to update the figure


# After training loop finishes, save the final plot
if plot:
    plt.savefig(f"{save_path}/final_loss_plot.png")

print("Training completed!")
    