#!/usr/bin/env python

import torch

from model import Img2Tex as Model
from model import device

def train():
    model = Model().cuda()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction='sum')

    N = 2
    x = torch.randn(N, 1, 100, 100, device=device)
    y = torch.randn(N, 80, device=device)

    for epoch in range(0, 20):
        print('starting epoch', epoch)
        for batch in range(0, 100):
            y_pred = model(x)

            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

if __name__ == '__main__':
    train()
