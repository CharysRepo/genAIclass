import torch
from neural_net import *
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        X = np.array(self.df.x, dtype=np.float32).reshape(-1, 1)
        self.X = torch.from_numpy(X)
        y = np.array(self.df.y, dtype=np.float32).reshape(-1, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model, epochs, batch_size, train_dataset, val_dataset):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            # forward pass
            output = model(input_batch)
            loss = F.mse_loss(output, target_batch)
            train_loss += loss.item()

            # backward pass
            optimizer.zero_grad()  # Clear gradients w.r.t. parameters
            loss.backward()  # Getting gradients w.r.t. parameters
            optimizer.step()  # Updating parameters

        val_loss = 0.0
        model.eval()        
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                output = model(input_batch)
                val_loss += F.mse_loss(output, target_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"epoch: {epoch+1}, train_loss: {train_loss}, val_loss: {val_loss}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "data"
    train_dataset = CustomDataset(base_dir / "linear/train.csv")
    val_dataset = CustomDataset(base_dir / "linear/test.csv")

    model = RegressionNetwork(1, 1)
    batch_size = 10
    epochs = 100

    train(model, epochs, batch_size, train_dataset, val_dataset)
    print(model.state_dict())
    torch.save(model, base_dir / "models/model1.pth")