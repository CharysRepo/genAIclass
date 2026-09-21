import torch
import torch.nn as nn

torch.manual_seed(123)
vocab_size=100
embed = nn.Embedding(vocab_size, embedding_dim=5)
print(embed.state_dict())

inp_ids = torch.tensor([3, 2, 8])
output = embed(inp_ids)
print(output, output.shape)

inp_ids = torch.tensor([[3, 2, 8], [1, 4, 5]])
output = embed(inp_ids)
print(output, output.shape)
