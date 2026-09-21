import torch

# tensor from a scalar value
scalar_tensor = torch.tensor(7)
print(scalar_tensor)
print(type(scalar_tensor))
print(scalar_tensor.ndim)
print(scalar_tensor.shape)
print(scalar_tensor.device)
print(scalar_tensor.dtype)
print(scalar_tensor.item())

# tensor from a 1d-list
tmp = [1, 2, 3]
v = torch.tensor(tmp)
print(v)
print(v.ndim)
print(v.shape)
print(v.device)
print(v.tolist())

# tensor from 2-d list
tmp = [[1, 2, 3], [4, 5, 6]]
m = torch.tensor(tmp)
print(m)
print(m.ndim)
print(m.shape)
print(m.device)
print(m.tolist())

# tensor from 3-d list
tmp = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
t = torch.tensor(tmp)
print(t)
print(t.ndim)
print(t.shape)
print(t.device)
print(t.tolist())

# create a random tensors
torch.manual_seed(100)
r = torch.rand(2)
print(r)
rn = torch.randn(2)
print(rn)

x = torch.randn(3, 4)
print(x)

# create special tensors
z = torch.ones(2, 2)
print(z)

# create a tensor with values in a specific range
tensor_in_range = torch.arange(start=0, end=10, step=2)
print(tensor_in_range)

linspace_tensor = torch.linspace(start=0, end=10, steps=5)
print(linspace_tensor)

# create tensors with specific datatype
tensor_float16 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float16)
print(tensor_float16)
