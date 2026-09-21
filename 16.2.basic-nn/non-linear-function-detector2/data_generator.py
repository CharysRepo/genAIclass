from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# 1. Setup Relative Paths for CSV files
base_dir = Path(__file__).resolve().parent.parent / "data"
train_file = base_dir / "non-linear2/train.csv"
test_file = base_dir / "non-linear2/test.csv"

# 2. Generate Synthetic 3-Variable Non-Linear Data
torch.manual_seed(42)

# Generate 500 samples for 3 separate features (X1, X2, X3)
X1 = torch.linspace(-3, 3, 500).view(-1, 1)
X2 = torch.linspace(-2, 2, 500).view(-1, 1)
X3 = torch.linspace(-1, 1, 500).view(-1, 1)

# Combine them into a single tensor of shape (500, 3)
X = torch.cat([X1, X2, X3], dim=1)

# Define a 3-variable non-linear function
# Example equation: Y = (X1^2) + sin(X2) - exp(X3) + noise
Y = (X[:, 0:1].pow(2) + 
     torch.sin(X[:, 1:2]) - 
     torch.exp(X[:, 2:3]) + 
     torch.randn(X1.size()) * 0.2)

# 3. Convert PyTorch Tensors to NumPy for Splitting
X_np = X.numpy()
Y_np = Y.numpy()

# 4. Split Data (80% Train, 20% Test)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_np, Y_np, test_size=0.2, random_state=42, shuffle=True
)

# 5. Create DataFrames and Save to CSV
train_df = pd.DataFrame({
    "x1": X_train[:, 0], 
    "x2": X_train[:, 1], 
    "x3": X_train[:, 2], 
    "y": Y_train.flatten()
})

test_df = pd.DataFrame({
    "x1": X_test[:, 0], 
    "x2": X_test[:, 1], 
    "x3": X_test[:, 2], 
    "y": Y_test.flatten()
})

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)