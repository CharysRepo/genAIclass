import torch

if torch.cuda.is_available():
    print("CUDA is available.")

x = torch.randn(3, 4)
print(x)
print(x.device)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
print(x.device)