from pathlib import Path
import torch

base_dir = Path(__file__).resolve().parent.parent / "data"
model = torch.load(base_dir / "models/model1.pth", weights_only=False)
model.eval()
with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = torch.Tensor([[1.0], [2.0], [3.0]])
    input_batch = input_batch.to(device)
    predictions = model(input_batch)
print(predictions)