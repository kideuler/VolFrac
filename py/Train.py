import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import getopt, sys
import os

class VolFracDataSet(Dataset):
    def __init__(self, filename):
        xy = np.loadtxt(filename, delimiter=' ', dtype=np.float32)
        self.x = torch.from_numpy(xy[:, :5].astype(np.float32))
        self.y = torch.from_numpy(xy[:, [-1]].astype(np.float32))
        
        # Split into train/val/test
        x_temp, self.x_test, y_temp, self.y_test = train_test_split(self.x, self.y, test_size=0.2)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_temp, y_temp, test_size=0.2)
        
        # Normalize inputs
        self.x_mean = self.x_train.mean(dim=0)
        self.x_std = self.x_train.std(dim=0)
        self.x_train = (self.x_train - self.x_mean) / self.x_std
        self.x_val = (self.x_val - self.x_mean) / self.x_std
        self.x_test = (self.x_test - self.x_mean) / self.x_std
        
        # Normalize targets
        self.y_mean = self.y_train.mean()
        self.y_std = self.y_train.std()
        self.y_train = (self.y_train - self.y_mean) / self.y_std
        self.y_val = (self.y_val - self.y_mean) / self.y_std
        self.y_test = (self.y_test - self.y_mean) / self.y_std
        
        self.n_samples = self.x_train.shape[0]

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    
    def get_test_data(self):
        return self.x_test, self.y_test

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        # Larger architecture
        H1 = 8
        H2 = 8
        H3 = 8
        H4 = 8
        self.network = nn.Sequential(
            nn.Linear(input_size, H1),
            nn.BatchNorm1d(H1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(H2, H3),
            nn.BatchNorm1d(H3),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(H3, H4),
            nn.BatchNorm1d(H4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(H4, num_classes)
        )
        
        # Better initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He/Kaiming initialization
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x):
        return self.network(x)

# Normalization Module
class NormalizationModule(torch.nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_std', x_std)
        self.register_buffer('y_mean', torch.tensor([y_mean]))
        self.register_buffer('y_std', torch.tensor([y_std]))


### Beginning of Script

# Parse command line arguments
argumentList = sys.argv[1:]
options = "hd:s:m:n:"
long_options = ["help","data-path=", "save-path=","model-path=","num-epochs="]

# checking each argument
plot = False
path = 'build/data/VolFracData.dat'
save_path = 'models/'
load_model = False
model_path = 'models/model_weights.pth'
save_frequency = 0
num_epochs = 200

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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
n_samples, n_features = dataset.x_train.shape


model = NeuralNet(input_size=n_features, num_classes=1)
learning_rate = 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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

for epoch in range(num_epochs):
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
            