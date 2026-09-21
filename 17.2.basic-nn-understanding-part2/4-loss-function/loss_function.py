import torch
import torch.nn as nn

pred = torch.randn(3)
print("pred: ", pred)
target = torch.randn(3)
print("target: ", target)

mse_loss = nn.MSELoss()
loss = mse_loss(pred, target)
print("mse_loss: ", loss)
