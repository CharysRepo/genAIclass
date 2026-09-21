import torch
import torch.nn as nn

torch.manual_seed(100)

drop_out = nn.Dropout(p=0.5)
print(drop_out.training)
print(drop_out.state_dict())

drop_out.training = False
input = torch.randn(3, 5)
print(input, input.shape)
output = drop_out(input)
print(output, output.shape)
