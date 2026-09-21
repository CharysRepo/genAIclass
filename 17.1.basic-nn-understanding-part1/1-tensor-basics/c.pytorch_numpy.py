import numpy as np
import torch

np_array = np.ones((2, 2))
print(type(np_array))
t1 = torch.tensor(np_array)
print(t1)
t2 = torch.from_numpy(np_array)
print(t2)

torch_tensor = torch.ones(2, 2)
torch_to_numpy = torch_tensor.numpy()
print(torch_to_numpy)
print(type(torch_to_numpy))
