import torch
import torch.nn.functional as F

logits = torch.tensor([2.5, 0.1, -1.0])
targets = torch.tensor([1], dtype=torch.long)
loss = F.cross_entropy(logits, targets)
print(f"Calculated Functional Loss: {loss.item():.4f}")

logits = torch.tensor([
    [2.5, 0.1, -1.0], 
    [0.2, 0.5,  3.1] 
])
targets = torch.tensor([0, 2], dtype=torch.long)
loss = F.cross_entropy(logits, targets)
print(f"Calculated Functional Loss: {loss.item():.4f}")

