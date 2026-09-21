import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
print(logits)
predicted_idx = torch.argmax(logits, dim=-1).item()
print(predicted_idx)
probabilities = F.softmax(logits, dim=-1)
print(probabilities)
predicted_idx = torch.argmax(probabilities, dim=-1).item()
print(predicted_idx)