import torch.nn as nn
import torch
from torchinfo import summary

class RegressionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim1)
        self.hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        print(x, x.shape)
        x = self.input(x)
        print(x, x.shape)
        x = self.hidden(x)
        print(x, x.shape)
        x = self.output(x)
        print(x, x.shape)
        return x


if __name__ == "__main__":
    model = RegressionNetwork(2, 3, 5, 1)
    # model inference
    inp = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    #inp = torch.tensor([0.1, 0.2])
    print(model(inp))

    # model info(external library)
    print(model)
    summary(model)

    # model weights (all)
    print(model.state_dict())

    # model weights (iterative)
    for name, param in model.named_parameters():
        print(name, param.data.shape, param.data)
