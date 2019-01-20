#!/usr/bin/env python

import torch

from model import Img2Tex as Model

def train():
    model = Model()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction='sum')

    x = torch.randn(10, 10)
    y = torch.randn(10, 10)

    for i in range(0, 100):
        y_pred = model(x)

        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        loss.backward()

        optimizer.step()
        print(i)

if __name__ == '__main__':
    train()
