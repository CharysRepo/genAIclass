from pathlib import Path
import torch

base_dir = Path(__file__).resolve().parent.parent / "data"
model = torch.load(base_dir / "models/model3.pth", weights_only=False)
model.eval()
with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = torch.tensor([
        [1.0, 2.0, 3.0],  # Sample 1
        [4.0, 5.0, 6.0],  # Sample 2
        [7.0, 8.0, 9.0]   # Sample 3
    ], dtype=torch.float32)
    input_batch = input_batch.to(device)
    predictions = model(input_batch)
print(predictions)