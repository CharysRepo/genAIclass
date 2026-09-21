import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split


class RegressionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.linear.weight = nn.Parameter(torch.tensor([[-4.0]]))
        self.linear.bias = nn.Parameter(torch.tensor([[-4.0]]))

    def forward(self, x):
        out = self.linear(x)
        return out

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.reshape(-1,1))
        self.y = torch.from_numpy(y.reshape(-1, 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def plot_losses(epochs, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(10, 10))

    # Plot training and validation loss against epochs
    ax1.plot(epochs, train_losses, label="Training loss")
    ax1.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    plt.show()
    plt.close()

from matplotlib import colormaps 

def plot_loss_surface_with_bias_trajectory(train_dataset, weights_history, biases_history, losses_history):
    all_X = train_dataset.X
    all_Y = train_dataset.y
    
    # 1. Generate a 2D coordinate grid for both Weights and Biases
    w_grid = np.linspace(-10.0, 10.0, 100)
    b_grid = np.linspace(-10.0, 10.0, 100)
    W, B = np.meshgrid(w_grid, b_grid)
    Z_losses = np.zeros_like(W)
    
    # Calculate global dataset MSE loss over every coordinate combination
    for i in range(len(w_grid)):
        for j in range(len(b_grid)):
            predictions = all_X * W[j, i] + B[j, i]
            loss = F.mse_loss(predictions, all_Y)
            Z_losses[j, i] = loss.item()
            
    # 2. Convert history variables into arrays for easier slicing and plotting
    w_hist = np.array(weights_history)
    b_hist = np.array(biases_history)
    loss_hist = np.array(losses_history)

    # 3. Create a Side-by-Side Canvas
    fig = plt.figure(figsize=(18, 8))
    
    # ==================== LEFT SUBPLOT: 3D SURFACE BOWL ====================
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot smooth continuous 3D wireframe mesh bowl surface
    # stride control regulates mesh line density
    surf = ax1.plot_surface(W, B, Z_losses, cmap='viridis', alpha=0.6, 
                            linewidth=0.5, edgecolors='black', rstride=5, cstride=5)
    
    # Draw continuous 3D trajectory line flying above/resting on the landscape bowl surface
    ax1.plot(w_hist, b_hist, loss_hist, color='red', linewidth=3, zorder=10, label='Optimization Path')
    
    # Overlay individual step optimization dots in 3D coordinate space
    ax1.scatter(w_hist, b_hist, loss_hist, color='cyan', s=40, edgecolors='black', alpha=1.0, zorder=11)
    
    # Highlight start and final states
    ax1.scatter(w_hist[0], b_hist[0], loss_hist[0], color='lime', s=150, edgecolors='black', zorder=12, label='Start State')
    ax1.scatter(w_hist[-1], b_hist[-1], loss_hist[-1], color='magenta', s=150, edgecolors='black', zorder=12, label='Final State')
    
    ax1.set_xlabel("Weight (w)")
    ax1.set_ylabel("Bias (b)")
    ax1.set_zlabel("Loss (MSE)")
    ax1.set_title("3D Loss Surface (Paraboloid Bowl)")
    ax1.legend()
    # Adjust camera viewing angle (Elevation, Azimuth) for optimal clarity
    ax1.view_init(elev=25, azim=-60)

    # ==================== RIGHT SUBPLOT: 2D CONTOUR MAP ====================
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Draw flat background loss surface contours
    contour = ax2.contourf(W, B, Z_losses, levels=30, cmap='viridis', alpha=0.85)
    ax2.contour(W, B, Z_losses, levels=30, colors='black', alpha=0.15, linewidths=0.5)
    
    # Overlay the training path trajectory across the landscape flatness
    ax2.plot(w_hist, b_hist, color='white', linestyle='-', linewidth=2, zorder=3)
    ax2.scatter(w_hist, b_hist, color='cyan', s=40, edgecolors='black', zorder=4, label='GD Steps')
    
    # Highlight start and final states
    ax2.scatter(w_hist[0], b_hist[0], color='lime', s=150, edgecolors='black', zorder=5, label='Start State')
    ax2.scatter(w_hist[-1], b_hist[-1], color='magenta', s=150, edgecolors='black', zorder=5, label='Final State')
    
    ax2.set_xlabel("Weight Parameter Value (w)")
    ax2.set_ylabel("Bias Parameter Value (b)")
    ax2.set_title("Top-Down 2D Contour Map View")
    ax2.legend(loc='upper right')
    
    # Global layout configurations
    fig.colorbar(contour, ax=ax2, label='Mean Squared Error (Loss)')
    fig.suptitle("Gradient Descent: 3D Surface vs 2D Contour Trajectory Comparison", fontsize=16, weight='bold')
    fig.tight_layout()
    
    plt.show()
    plt.close()
    

def train(model, epochs, batch_size, train_dataset, val_dataset):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_losses = []
    val_losses = []
    weight_history = []
    bias_history = []
    step_losses = []
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            # clear gradients of parameters from previous step
            optimizer.zero_grad() 

            # compute neural net output (forward pass)
            output = model(input_batch)

            # compute loss
            loss = F.mse_loss(output, target_batch)
            train_loss += loss.item()
            step_losses.append(loss.item())
            weight_history.append(model.linear.weight.item())
            bias_history.append(model.linear.bias.item())

            # compute the slope of loss curve at this weights (backward pass)
            loss.backward()

            # update parameters based on learning rate and calculated slopes
            optimizer.step()

        val_loss = 0.0
        model.eval()        
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                output = model(input_batch)
                val_loss += F.mse_loss(output, target_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"epoch: {epoch+1}, train_loss: {train_loss}, val_loss: {val_loss}")
    return train_losses, val_losses, step_losses, weight_history, bias_history

if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.linspace(-3, 3, 500).view(-1, 1)  # 500 points between -3 and 3
    Y = 2 * X + 1.0 + torch.randn(X.size()) * 0.2 # Linear function: Y = 2X + 1 + Gaussian noise

    print(X)
    print(Y)

    X_np = X.numpy()
    Y_np = Y.numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_np, Y_np, test_size=0.2, random_state=42, shuffle=True
    )
    train_dataset = CustomDataset(X_train, Y_train)
    val_dataset = CustomDataset(X_test, Y_test)

    model = RegressionNetwork(1, 1)
    batch_size = len(train_dataset)
    epochs = 50

    train_losses, val_losses, step_losses, weight_history, bias_history = train(model, epochs, batch_size, train_dataset, val_dataset)

    plot_losses(list(range(1, epochs + 1)), train_losses, val_losses)
    plot_loss_surface_with_bias_trajectory(train_dataset, weight_history, bias_history, step_losses)

    print(model.state_dict())