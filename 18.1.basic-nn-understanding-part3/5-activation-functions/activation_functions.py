import torch
import torch.nn.functional as F

data = torch.randn(2)
print(data)
print(F.relu(data))

data = torch.randn(4)
print(data)
print(F.relu(data))

data = torch.randn(4)
print(data)
print(F.gelu(data))

data = torch.tensor([-2, -1, 1, 2])
print(data)
print(F.sigmoid(data))

data = torch.tensor([-2, -1, 1, 2])
print(data)
print(F.tanh(data))


