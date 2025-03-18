import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import getopt, sys
import os
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import time
import math

class AreaPredictionNet(nn.Module):
    def __init__(self, input_size=4, hidden_sizes=[32, 64, 32], num_classes=1):
        """
        Neural network for predicting area after a line cut
        
        Args:
            input_size (int): Size of input (point_x, point_y, normal_x, normal_y)
            hidden_sizes (list): List of hidden layer sizes
            num_classes (int): Size of output (area)
        """
        super(AreaPredictionNet, self).__init__()
        
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
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Final output layer (no activation as we're predicting a continuous value)
        x = self.output_layer(x)
        return x

class AreaDataset(Dataset):
    def __init__(self, num_samples=10000):
        """
        Generate synthetic data for area prediction
        
        Args:
            num_samples (int): Number of samples to generate
        """
        # Generate random points and normal vectors
        np.random.seed(42)
        points = np.random.rand(num_samples, 2)  # Points in [0,1]x[0,1]
        
        # Generate random angles and convert to normal vectors
        angles = np.random.uniform(0, 2*np.pi, num_samples)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Normalize the normals
        norms = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2)
        normals = normals / norms[:, np.newaxis]
        
        # Combine points and normals into input features
        self.x = np.column_stack((points, normals)).astype(np.float32)
        
        # Calculate exact areas (see calculate_area function)
        self.y = np.array([self.calculate_area(p[0], p[1], n[0], n[1]) 
                          for p, n in zip(points, normals)]).astype(np.float32).reshape(-1, 1)
        
        # Split into train and validation
        indices = np.random.permutation(num_samples)
        split = int(0.8 * num_samples)
        train_idx, val_idx = indices[:split], indices[split:]
        
        self.x_train = torch.tensor(self.x[train_idx], dtype=torch.float32)
        self.y_train = torch.tensor(self.y[train_idx], dtype=torch.float32)
        self.x_val = torch.tensor(self.x[val_idx], dtype=torch.float32)
        self.y_val = torch.tensor(self.y[val_idx], dtype=torch.float32)
        
        # Compute mean and std for normalization
        self.x_mean = torch.mean(self.x_train, dim=0)
        self.x_std = torch.std(self.x_train, dim=0)
        self.y_mean = torch.mean(self.y_train)
        self.y_std = torch.std(self.y_train)
        
        # Apply normalization
        self.x_train = (self.x_train - self.x_mean) / self.x_std
        self.x_val = (self.x_val - self.x_mean) / self.x_std
        self.y_train = (self.y_train - self.y_mean) / self.y_std
        self.y_val = (self.y_val - self.y_mean) / self.y_std
        
        self.n_samples = len(self.x_train)
        
        print(f"Dataset created with {self.n_samples} training samples and {len(self.x_val)} validation samples")
        print(f"X mean: {self.x_mean}, X std: {self.x_std}")
        print(f"Y mean: {self.y_mean.item()}, Y std: {self.y_std.item()}")

    def calculate_area(self, point_x, point_y, normal_x, normal_y):
        """
        Calculate the area of the portion of the unit square that lies on the positive
        side of the line with normal vector (normal_x, normal_y) passing through point (point_x, point_y)
        
        The line equation is: normal_x * (x - point_x) + normal_y * (y - point_y) = 0
        The positive side is where normal_x * (x - point_x) + normal_y * (y - point_y) >= 0
        """
        # Calculate the line equation: ax + by + c = 0
        a = normal_x
        b = normal_y
        c = -(a * point_x + b * point_y)
        
        # Function to determine if a point is on the positive side
        def is_positive(x, y):
            return a*x + b*y + c >= 0
        
        # Check all four corners
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        positive_corners = [is_positive(x, y) for x, y in corners]
        
        # If all corners are positive or all negative, return 1 or 0
        if all(positive_corners):
            return 1.0
        if not any(positive_corners):
            return 0.0
        
        # Calculate intersection points with the unit box boundaries
        intersections = []
        
        # Check horizontal boundaries (y = 0 and y = 1)
        if abs(b) > 1e-10:  # Avoid division by zero
            for y_val in [0, 1]:
                x_val = -(b * y_val + c) / a
                if 0 <= x_val <= 1:
                    intersections.append((x_val, y_val))
                    
        # Check vertical boundaries (x = 0 and x = 1)
        if abs(a) > 1e-10:  # Avoid division by zero
            for x_val in [0, 1]:
                y_val = -(a * x_val + c) / b
                if 0 <= y_val <= 1:
                    intersections.append((x_val, y_val))
        
        # Need to handle special cases (line parallel to axes)
        if len(intersections) < 2:
            # This can happen if the line passes exactly through corners
            # or if it's parallel to an axis
            if abs(a) < 1e-10:  # Line is horizontal
                y_val = -c / b
                intersections = [(0, y_val), (1, y_val)]
            elif abs(b) < 1e-10:  # Line is vertical
                x_val = -c / a
                intersections = [(x_val, 0), (x_val, 1)]
        
        # Sort points for polygon area calculation
        # First, find positive corners
        positive_corner_points = [corners[i] for i in range(4) if positive_corners[i]]
        
        # Create the polygon by combining positive corners and intersection points
        polygon = positive_corner_points + intersections
        
        # Sort polygon points in counter-clockwise order
        # Calculate centroid as reference point
        centroid_x = sum(p[0] for p in polygon) / len(polygon)
        centroid_y = sum(p[1] for p in polygon) / len(polygon)
        
        # Sort based on angle from centroid
        polygon.sort(key=lambda p: math.atan2(p[1] - centroid_y, p[0] - centroid_x))
        
        # Calculate polygon area using the Shoelace formula
        area = 0.0
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        area = abs(area) / 2.0
        
        return area

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

class NormalizationModule(torch.nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std):
        super(NormalizationModule, self).__init__()
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_std', x_std)
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_std', y_std)
    
    def forward(self, x):
        # Normalize input
        x_normalized = (x - self.x_mean) / self.x_std
        return x_normalized
    
    def unnormalize_output(self, y_normalized):
        # Convert normalized output back to original scale
        return y_normalized * self.y_std + self.y_mean

def train_model(dataset, model, num_epochs=200, batch_size=64, learning_rate=0.01, plot=True):
    """
    Train the neural network model
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    criterion = nn.MSELoss()
    
    # Training tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    # Create normalization module
    norm_module = NormalizationModule(
        dataset.x_mean,
        dataset.x_std,
        dataset.y_mean,
        dataset.y_std
    )
    
    # Setup plotting
    if plot:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_train_loss += loss.item()
            num_batches += 1
        
        # Calculate average training loss
        avg_train_loss = running_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(dataset.x_val)
            val_loss = criterion(val_pred, dataset.y_val).item()
            val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "models/best_area_model.pth")
                torch.jit.script(model).save("models/AreaPredictor.pt")
                torch.jit.script(norm_module).save("models/area_normalization.pt")
                save_path = "models"
                Write_model(save_path + "/model.dat", norm_module, dataset, model)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Update plot every 10 epochs
        if plot and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1):
            epochs_range = list(range(1, len(train_losses) + 1))
            
            ax1.cla()
            ax2.cla()
            
            ax1.plot(epochs_range, train_losses, 'b-')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            ax2.plot(epochs_range, val_losses, 'r-')
            ax2.set_title('Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
            
            # Mark best validation loss
            best_epoch = val_losses.index(min(val_losses)) + 1
            ax2.plot(best_epoch, min(val_losses), 'go', markersize=8, label=f"Best: {min(val_losses):.6f}")
            ax2.legend()
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save final plot
    if plot:
        plt.savefig("models/area_training_plot.png")
        plt.ioff()
        plt.show()
    
    # Load best model
    model.load_state_dict(torch.load("models/best_area_model.pth"))
    
    return model, norm_module

def evaluate_model(model, norm_module, num_samples=20):
    """
    Evaluate the model on random samples and visualize predictions
    """
    # Generate random samples
    np.random.seed(123)  # Different seed for evaluation
    points = np.random.rand(num_samples, 2)
    angles = np.random.uniform(0, 2*np.pi, num_samples)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Normalize the normals
    norms = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2)
    normals = normals / norms[:, np.newaxis]
    
    # Prepare input
    x = np.column_stack((points, normals)).astype(np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Calculate exact areas for comparison
    dataset = AreaDataset(1)  # Just using this to access calculate_area method
    exact_areas = np.array([dataset.calculate_area(p[0], p[1], n[0], n[1]) 
                           for p, n in zip(points, normals)])
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        # Normalize inputs
        x_normalized = norm_module(x_tensor)
        # Get normalized predictions
        y_normalized = model(x_normalized)
        # Unnormalize predictions
        predictions = norm_module.unnormalize_output(y_normalized).numpy().flatten()
    
    # Compute errors
    errors = np.abs(predictions - exact_areas)
    mae = np.mean(errors)
    mape = np.mean(errors / np.maximum(exact_areas, 1e-8)) * 100
    
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Visualize predictions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(exact_areas, predictions, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Exact Area')
    plt.ylabel('Predicted Area')
    plt.title('Prediction vs. Exact Area')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(exact_areas, errors, alpha=0.7)
    plt.xlabel('Exact Area')
    plt.ylabel('Absolute Error')
    plt.title(f'Errors (MAE={mae:.6f}, MAPE={mape:.2f}%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("models/area_evaluation.png")
    plt.show()
    
    # Visualize some examples
    fig, axs = plt.subplots(4, 5, figsize=(15, 12))
    axs = axs.flatten()
    
    for i in range(min(num_samples, 20)):
        point_x, point_y = points[i]
        normal_x, normal_y = normals[i]
        exact_area = exact_areas[i]
        predicted_area = predictions[i]
        
        # Plot unit square
        axs[i].plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-')
        
        # Calculate the line
        # normal_x * (x - point_x) + normal_y * (y - point_y) = 0
        # normal_x * x + normal_y * y = normal_x * point_x + normal_y * point_y
        # y = (-normal_x * x + normal_x * point_x + normal_y * point_y) / normal_y
        
        if abs(normal_y) > 1e-10:
            # Not a vertical line
            x_vals = np.linspace(-0.5, 1.5, 100)
            c = normal_x * point_x + normal_y * point_y
            y_vals = (-normal_x * x_vals + c) / normal_y
            axs[i].plot(x_vals, y_vals, 'r-')
        else:
            # Vertical line
            y_vals = np.linspace(-0.5, 1.5, 100)
            x_vals = np.ones_like(y_vals) * point_x
            axs[i].plot(x_vals, y_vals, 'r-')
        
        # Plot the point
        axs[i].plot(point_x, point_y, 'ro', markersize=8)
        
        # Plot normal vector (scaled for visibility)
        arrow_scale = 0.2
        axs[i].arrow(point_x, point_y, arrow_scale * normal_x, arrow_scale * normal_y,
                    head_width=0.05, head_length=0.05, fc='r', ec='r')
        
        # Set limits and title
        axs[i].set_xlim(-0.1, 1.1)
        axs[i].set_ylim(-0.1, 1.1)
        axs[i].set_title(f"Exact: {exact_area:.4f}, Pred: {predicted_area:.4f}")
        axs[i].set_aspect('equal')
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.savefig("models/area_examples.png")
    plt.show()


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

def main():
    """
    Main function to train and evaluate the model
    """
    # Parse command line arguments
    plot = False
    num_samples = 50000
    num_epochs = 200
    batch_size = 64
    learning_rate = 0.01
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hpn:e:b:l:", 
                               ["help", "plot", "num-samples=", "epochs=", "batch-size=", "learning-rate="])
    except getopt.GetoptError as err:
        print(str(err))
        print("python Train_LineArea.py [-p] [-n num_samples] [-e epochs] [-b batch_size] [-l learning_rate]")
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("python Train_LineArea.py [-p] [-n num_samples] [-e epochs] [-b batch_size] [-l learning_rate]")
            sys.exit()
        elif opt in ("-p", "--plot"):
            plot = True
        elif opt in ("-n", "--num-samples"):
            num_samples = int(arg)
        elif opt in ("-e", "--epochs"):
            num_epochs = int(arg)
        elif opt in ("-b", "--batch-size"):
            batch_size = int(arg)
        elif opt in ("-l", "--learning-rate"):
            learning_rate = float(arg)
    
    print("Starting Area Prediction Neural Network Training...")
    print(f"Parameters: samples={num_samples}, epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
    
    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create dataset
    dataset = AreaDataset(num_samples=num_samples)
    
    # Create model
    model = AreaPredictionNet(input_size=4, hidden_sizes=[64, 128, 256, 128, 64], num_classes=1)
    
    # Train model
    model, norm_module = train_model(
        dataset, 
        model, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        plot=plot
    )
    
    # Evaluate model
    evaluate_model(model, norm_module)

    # Write model to file
    
    
    print("Training complete! Models saved to 'models/' directory.")

if __name__ == "__main__":
    main()