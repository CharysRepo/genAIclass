import torch
import torch.nn as nn

token_1 = torch.tensor([142.3, -89.1, 12.4, 305.0])
token_2 = torch.tensor([0.1, -0.4, 0.3, -0.2])
input = torch.stack([token_1, token_2])
print(input, input.shape)

layer_norm = nn.LayerNorm(4, elementwise_affine=False)
print(layer_norm.state_dict())
output = layer_norm(input)
print(output, output.shape)

print(f"Row 1 Mean: {output[0].mean().item():.4f}, Variance: {output[0].var(correction=0).item():.4f}")
print(f"Row 2 Mean: {output[1].mean().item():.4f}, Variance: {output[1].var(correction=0).item():.4f}")
