from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

base_dir = Path(__file__).resolve().parent.parent / "data"
train_file = base_dir / "non-linear1/train.csv"
test_file = base_dir / "non-linear1/test.csv"

torch.manual_seed(42)
X = torch.linspace(-3, 3, 500).view(-1, 1)  # 500 points between -3 and 3
Y = X.pow(2) + torch.randn(X.size()) * 0.2  # Quadratic function with added Gaussian noise

X_np = X.numpy()
Y_np = Y.numpy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X_np, Y_np, test_size=0.2, random_state=42, shuffle=True
)

train_df = pd.DataFrame({"x": X_train.flatten(), "y": Y_train.flatten()})
test_df = pd.DataFrame({"x": X_test.flatten(), "y": Y_test.flatten()})

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)
