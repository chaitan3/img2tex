#!/usr/bin/env python

import torch
import time

from model import Img2Tex as Model
from model import device
from data import load_images

N = 2
n_epochs = 20
learning_rate= 0.1

def train():
    model = Model().cuda()
    criterion = torch.nn.NLLLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    train_data, _ = load_images()

    for epoch in range(0, n_epochs):
        print('starting epoch', epoch)
        for key, batch in train_data.items():
            batch_size = len(batch[0])
            print(key, batch[1].shape)
            continue
            for i in range(0, batch_size, N):
                start = time.time()

                x = batch[0][i:i+N]
                y = batch[1][i:i+N]
                y_pred = model(x)

                optimizer.zero_grad()
                loss = criterion(y_pred, y)
                loss.backward()

                optimizer.step()
                end = time.time()

                print('epoch: {} {} {}/{}, loss: {}'.format(epoch, key, i, batch_size, loss.item()))
                print('time step:', end-start)

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
