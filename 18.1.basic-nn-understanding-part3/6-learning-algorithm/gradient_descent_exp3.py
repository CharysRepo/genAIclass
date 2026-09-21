import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from matplotlib import colormaps 
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

class RegressionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        out = self.linear2(x)
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

def plot_loss_surface_with_relu_trajectory(train_dataset, w1_history, w2_history, losses_history):
    all_X = train_dataset.X
    all_Y = train_dataset.y
    
    # 1. Generate a 2D coordinate grid for both Weights and Biases
    w1_grid = np.linspace(-10.0, 10.0, 150)
    w2_grid = np.linspace(-10.0, 10.0, 150)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    Z_losses = np.zeros_like(W1)
    
    model.cpu()
    with torch.no_grad():
        # Keep a safe backup of optimized weights to restore them post-calculation
        original_w1 = model.linear1.weight.clone()
        original_w2 = model.linear2.weight.clone()
        
        for i in range(len(w1_grid)):
            for j in range(len(w2_grid)):
                # Inject grid combinations into the first index slot of the weights
                model.linear1.weight[0, 0] = W1[j, i]
                model.linear2.weight[0, 0] = W2[j, i]
                
                predictions = model(all_X)
                loss = F.mse_loss(predictions, all_Y)
                Z_losses[j, i] = loss.item()
                
        # Restore true parameters to the network structure
        model.linear1.weight.copy_(original_w1)
        model.linear2.weight.copy_(original_w2)
            
    w1_hist = np.array(w1_history)
    w2_hist = np.array(w2_history)
    loss_hist = np.array(losses_history)

    fig = plt.figure(figsize=(18, 8))
    
    # ==================== LEFT SUBPLOT: 3D JAGGED SURFACE ====================
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(W1, W2, Z_losses, cmap='viridis', alpha=0.6, 
                            linewidth=0.2, edgecolors='black', rstride=4, cstride=4)
    
    ax1.plot(w1_hist, w2_hist, loss_hist, color='red', linewidth=3, zorder=10, label='Optimization Path')
    ax1.scatter(w1_hist, w2_hist, loss_hist, color='cyan', s=40, edgecolors='black', zorder=11)
    ax1.scatter(w1_hist, w2_hist, loss_hist, color='lime', s=150, edgecolors='black', zorder=12, label='Start State')
    ax1.scatter(w1_hist[-1], w2_hist[-1], loss_hist[-1], color='magenta', s=150, edgecolors='black', zorder=12, label='Final State')
    
    ax1.set_xlabel("Linear1 Weight Index [0,0]")
    ax1.set_ylabel("Linear2 Weight Index [0,0]")
    ax1.set_zlabel("Loss (MSE)")
    ax1.set_title("3D Non-Convex ReLU Loss Surface")
    ax1.legend()
    ax1.view_init(elev=30, azim=-135)

    # ==================== RIGHT SUBPLOT: 2D CONTOUR MAP ====================
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(W1, W2, Z_losses, levels=40, cmap='viridis', alpha=0.85)
    ax2.contour(W1, W2, Z_losses, levels=40, colors='black', alpha=0.15, linewidths=0.5)
    
    ax2.plot(w1_hist, w2_hist, color='white', linestyle='-', linewidth=2, zorder=3)
    ax2.scatter(w1_hist, w2_hist, color='cyan', s=40, edgecolors='black', zorder=4, label='GD Steps')
    ax2.scatter(w1_hist, w2_hist, color='lime', s=150, edgecolors='black', zorder=5, label='Start State')
    ax2.scatter(w1_hist[-1], w2_hist[-1], color='magenta', s=150, edgecolors='black', zorder=5, label='Final State')
    
    ax2.set_xlabel("Linear1 Weight Value")
    ax2.set_ylabel("Linear2 Weight Value")
    ax2.set_title("Top-Down 2D Contour View (Shows Non-Linear Fractures)")
    ax2.legend(loc='upper right')
    
    fig.colorbar(contour, ax=ax2, label='Mean Squared Error (Loss)')
    fig.suptitle("Gradient Descent Path on a Non-Convex ReLU Surface", fontsize=16, weight='bold')
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
    w1_history  = []
    w2_history  = []
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
            w1_history.append(model.linear1.weight[0, 0].item())
            w2_history.append(model.linear2.weight[0, 0].item())

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
    return train_losses, val_losses, step_losses, w1_history, w2_history

if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.linspace(-3, 3, 500).view(-1, 1)  # 500 points between -3 and 3
    Y = X.pow(2) + torch.randn(X.size()) * 0.2  # Quadratic function with added Gaussian noise
    print(X)
    print(Y)

    X_np = X.numpy()
    Y_np = Y.numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_np, Y_np, test_size=0.2, random_state=42, shuffle=True
    )
    train_dataset = CustomDataset(X_train, Y_train)
    val_dataset = CustomDataset(X_test, Y_test)

    model = RegressionNetwork(1, 1, 1)
    batch_size = len(train_dataset)
    epochs = 50

    train_losses, val_losses, step_losses, w1_history, w2_history = train(model, epochs, batch_size, train_dataset, val_dataset)

    plot_losses(list(range(1, epochs + 1)), train_losses, val_losses)
    plot_loss_surface_with_relu_trajectory(train_dataset, w1_history, w2_history, step_losses)

    print(model.state_dict())
