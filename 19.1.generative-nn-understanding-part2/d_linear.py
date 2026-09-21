import torch
import torch.nn as nn

torch.manual_seed(100)

lin = nn.Linear(in_features=4, out_features=2)
print(lin.state_dict())

input = torch.tensor([0.1, 0.2, 0.3, 0.4])
print(input, input.shape)
output = lin(input)
print(output, output.shape)

input = torch.tensor([[0.1, 0.2, 0.3, 0.15],[0.25, 0.4, 0.5, 0.6]])
print(input, input.shape)
output = lin(input)
print(output, output.shape)

