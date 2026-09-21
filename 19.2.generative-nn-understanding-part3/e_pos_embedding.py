import torch
from torch.nn import Embedding

torch.manual_seed(123)

pos_embed = Embedding(10, 5)
print(pos_embed.state_dict())

positions = torch.tensor([0, 2, 5])
print(positions)
output = pos_embed(positions)
print(output, output.shape)
