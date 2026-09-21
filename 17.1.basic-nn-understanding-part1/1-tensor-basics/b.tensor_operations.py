import torch

# indexing
print("Indexing")
x = torch.randn(3, 4)
print(x)
print(x[0])
print(x[0][0])

x = torch.randn(3)
print(x)
y = torch.randn(3)
print(y)

# addition
print("Addition")
z = x + y
print(z)
print(torch.add(x, y))

# multiplication
print("Multiplication")
z = x * y
print(z)
print(torch.mul(x, y))

# sum
print("Sum")
print(x.sum())
print(torch.sum(x))

# accessing the value of a tensor with single element
tmp = torch.sum(x)
print(tmp.item())

# mean
print("Mean")
print(x.mean())
print(torch.mean(x))

# standard deviation
print("SD")
print(x.std())
print(torch.std(x))

# concatenate rows
print("Concatenation")
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# concatenate columns
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1)  # second arg specifies which axis to concat along
print(z_2)

# reshape
print("Reshape")
x = torch.randn(2, 3)
print(x)
print(x.view(1, -1))
print(x.view(3, -1))
print(x.view(6))

x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))  # Reshape to 2 rows, 12 columns
print(x.view(2, -1))
print(x.view(3, -1))
print(x.view(24, -1))


