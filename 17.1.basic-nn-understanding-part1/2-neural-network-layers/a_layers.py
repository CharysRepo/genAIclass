import torch
import torch.nn as nn

torch.manual_seed(30)
data = torch.randn(5)
print(data)

# single neuron layer
lin = nn.Linear(5, 1)
print(lin.weight)
print(lin.bias)
print(lin(data))

# two neuron layer
lin = nn.Linear(5, 2)
print(lin.weight)
print(lin.bias)
print(lin(data))

device = "cuda" if torch.cuda.is_available() else "cpu"
data = data.to(device)
lin = lin.to(device)
print(lin(data))
