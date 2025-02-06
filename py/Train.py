import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import getopt, sys

### Creating Dataset Class
class VolFracDataSet(Dataset):
    def __init__(self, filename):
        xy = np.loadtxt(filename, delimiter=' ', dtype=np.float32)
        self.x = torch.from_numpy(xy[:, :5].astype(np.float32))
        self.y = torch.from_numpy(xy[:, [-1]].astype(np.float32))
        self.y = self.y.view(self.y.shape[0], 1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.n_samples = self.x_train.shape[0]

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    
    def get_test_data(self):
        return self.x_test, self.y_test


### Creating Neural Network Class   
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.l4 = nn.Linear(hidden_size_2, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l4(out)
        return out


### Beginning of Script

# Parse command line arguments
argumentList = sys.argv[1:]
options = "hd:s:m:Pf:n:"
long_options = ["help","data-path=", "save-path=","model-path=","plot", "save-frequency=", "num-epochs="]

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
            print ("python Train.py -p <data-path> -P -s <save-frequency>")
            print ("-d --data-path: path to the data file")
            print ("-s --save-path: path to save the model")
            print ("-m --model-path: path to load the model")
            print ("-P --plot: plot the training and testing loss")
            print ("-f --save-frequency: save the model every <save-frequency> epochs")
            print ("-n --num-epochs: number of epochs to train")
            sys.exit(0)
        if currentArgument in ("-d", "--data-path"):
            path = currentValue
        if currentArgument in ("-s", "--save-path"):
            save_path = currentValue
        if currentArgument in ("-m", "--model-path"):
            model_path = currentValue
            load_model = True
        elif currentArgument in ("-P", "--plot"):
            plot = True
        elif currentArgument in ("-f", "--save-frequency"):
            save_frequency = int(currentValue)
        elif currentArgument in ("-n", "--num-epochs"):
            num_epochs = int(currentValue)

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))

dataset = VolFracDataSet(path)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
n_samples, n_features = dataset.x_train.shape

input_size = n_features
output_size = 1
hidden_size = 7
hidden_size2 = 4
hidden_size3 = 2
model = NeuralNet(input_size, hidden_size, hidden_size2, output_size)

if load_model:
    model = torch.load(model_path, weights_only=False)
    model.eval()

learning_rate = 0.0005

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# 3) Training loop
if plot:
    plt.ion()
trainingEpoch_loss = []
testingEpoch_loss = []
epochs = []
for epoch in range(num_epochs):
    # Forward pass and loss
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):
        y_predicted = model(x_batch)
        loss = criterion(y_predicted, y_batch)
    
        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        y_test_predicted = model(dataset.x_test)
        loss_test = criterion(y_test_predicted, dataset.y_test)

        print(f'epoch: {epoch+1}, loss = {loss.item():.10f}, test loss = {loss_test.item():.10f}')
        epochs.append(epoch)
        trainingEpoch_loss.append(math.log10(loss.item()))
        testingEpoch_loss.append(math.log10(loss_test.item()))
        # plot the graph
        if plot:
            plt.clf()
            plt.plot(epochs, trainingEpoch_loss, label='Training Loss')
            plt.plot(epochs, testingEpoch_loss, label='Testing Loss')
            plt.legend()
            plt.show()
            plt.pause(0.01)

    if save_frequency > 0 and (epoch+1) % save_frequency == 0:
        print("Saving Model")
        example = torch.rand(1, 5)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(save_path+"/VolFrac.pt")
        torch.save(model.state_dict(), save_path+"/model_weights.pth")

# check loss on testing data
with torch.no_grad():
    X_test, y_test = dataset.get_test_data()
    y_predicted = model(X_test)
    loss = criterion(y_predicted, y_test)
    print(f'Loss: {loss:.10f}')
    example = torch.rand(1, 5)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(save_path+"/VolFrac.pt")
    torch.save(model.state_dict(), save_path+"/model_weights.pth")
    if plot:
        plt.ioff()
        plt.show()