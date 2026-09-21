import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
import time


class RegressionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.linear.weight = nn.Parameter(torch.tensor([[-4.0]]))

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

def plot_loss_surface_with_trajectory(train_dataset, weights_history, losses_history):
    # Extract all training data to compute the true surface background
    all_X = train_dataset.X
    all_Y = train_dataset.y
    
    # Generate a dense grid of possible weight values around your training path
    min_w = -6.0 
    max_w = 6.0
    weight_grid = np.linspace(min_w, max_w, 500)
    surface_losses = []
    
    # Calculate the exact mathematical loss for each grid weight value
    for w in weight_grid:
        predictions = all_X * w
        loss = F.mse_loss(predictions, all_Y)
        surface_losses.append(loss.item())
        
    fig, ax = plt.subplots(figsize=(12, 7))    
    ax.plot(weight_grid, surface_losses, color='black', alpha=0.3, linestyle='--', label='True Loss Surface')
    ax.plot(weights_history, losses_history, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.scatter(weights_history, losses_history, color='royalblue', s=40, zorder=4, label='GD Steps')
    
    ax.scatter(weights_history[0], losses_history[0], color='green', s=150, zorder=5, label='Start State')
    ax.scatter(weights_history[-1], losses_history[-1], color='red', s=150, zorder=5, label='Final State')
    
    ax.set_xlabel("Weight Parameter Value (w)")
    ax.set_ylabel("Mean Squared Error (Loss)")
    ax.set_title("Optimization Path Overlaid on Full Loss Surface")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    
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
    step_losses = []
    start_time = time.perf_counter()
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
    end_time = time.perf_counter()
    print(f"training time:{end_time-start_time} seconds")
    return train_losses, val_losses, step_losses, weight_history

if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.linspace(-3, 3, 100000).view(-1, 1)  # 500 points between -3 and 3
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
    batch_size = 30
    epochs = 10

    train_losses, val_losses, step_losses, weight_history = train(model, epochs, batch_size, train_dataset, val_dataset)

    plot_losses(list(range(1, epochs + 1)), train_losses, val_losses)
    plot_loss_surface_with_trajectory(train_dataset, weight_history, step_losses)

    print(model.state_dict())
