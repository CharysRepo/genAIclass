import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
print(logits)
probabilities = F.softmax(logits, dim=-1)
print(probabilities)
print(probabilities / 0.9)
print(probabilities / 0.1)
print(probabilities / 1.0)
print(probabilities / 5.0)

