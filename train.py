#!/usr/bin/env python

import torch

from model import Img2Tex as Model
from model import device
from data import load_images

learning_rate = 1e-4
batch_size = 16
n_epochs = 20
learning_rate= 0.1

def train():
    model = Model().cuda()
    criterion = torch.nn.NLLLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    train_data, _ = load_images()
    N = batch_size

    for epoch in range(0, n_epochs):
        print('starting epoch', epoch)
        for key, batch in train_data.items():
            for i in range(0, len(batch[0]), N):
                x = batch[0][i:i+N]
                y = batch[1][i:i+N]
                y_pred = model(x)

                optimizer.zero_grad()
                loss = criterion(y_pred, y)
                print('loss:', epoch, val_key, i/len(batch[0]), loss)
                loss.backward()

                optimizer.step()

            with torch.no_grad():
                loss = 0.
                n = 0.
                for val_key, val_batch in val_data.items():
                    for i in range(0, len(val_batch[0]), N):
                        x = val_batch[0][i:i+N]
                        y = val_batch[1][i:i+N]
                        y_pred = model(x)
                        n += 1
                        loss += criterion(y_pred, y)
                loss /= n
                print('validation loss:', loss)
                        
                    


if __name__ == '__main__':
    train()
