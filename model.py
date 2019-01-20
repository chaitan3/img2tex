import torch

class Img2Tex(torch.nn.Module):
    def __init__(self):
        super(Img2Tex, self).__init__()
        self.linear = torch.nn.Linear(10, 10)


    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
