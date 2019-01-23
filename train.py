#!/usr/bin/env python

import torch
import time

from model import Img2Tex as Model
from model import device
from data import load_images

N = 4
n_epochs = 20
learning_rate= 0.1

def model_size(model):
    return sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    #return sum(p.numel()*p.element_size() for p in model.parameters())

def save_checkpoint(n_samples, data, model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'data': data
        }, 'checkpoint_{}.pth'.format(n_samples))

def validation_loss(model, criterion, data):
    with torch.no_grad():
        loss = 0.
        n = 0.
        for key, batch in data.items():
            for i in range(0, len(batch[0]), N):
                x = batch[0][i:i+N].cuda()
                y = batch[1][i:i+N].cuda()
                y_pred = model(x)
                n += 1
                loss += criterion(y_pred, y)
                #print('val {} {}/{}, loss: {}'.format(key, i, len(batch[0]), loss.item()))
        loss /= n
    return loss

def load_checkpoint(n_samples, model, optimizer=None):
    checkpoint = torch.load('checkpoint_{}.pth'.format(n_samples))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['data']

def train():
    model = Model().cuda()
    criterion = torch.nn.NLLLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5**(epoch+1))

    start_epoch, start_key, _, _ = load_checkpoint(8240, model, optimizer)
    #start_epoch, start_key = 0, None, 0
    train_data, val_data = load_images(['train', 'validate'])

    data_keys = list(train_data.keys())
    if start_key:
        start_key = data_keys.index(start_key) + 1
    else:
        start_key = 0

    print('model parameter size (GB):', model_size(model)/1024**3)
    print(validation_loss(model, criterion, val_data))
    exit(1)

    n_samples = 0
    for epoch in range(start_epoch, n_epochs):
        print('starting epoch', epoch)
        scheduler.step()
        for key, batch in train_data.items():
            if data_keys.index(key) < start_key:
                continue
            batch_size = len(batch[0])
            #print(key, batch[1].shape)
            #continue
            for i in range(0, batch_size, N):
                start = time.time()

                #x = batch[0][i:i+N]
                #y = batch[1][i:i+N]
                x = batch[0][i:i+N].cuda()
                y = batch[1][i:i+N].cuda()
                y_pred = model(x)

                optimizer.zero_grad()
                loss = criterion(y_pred, y)
                loss.backward()

                optimizer.step()
                end = time.time()

                print('epoch: {} {} {}/{}, loss: {}'.format(epoch, key, i, batch_size, loss.item()))
                print('time step:', end-start)
                n_samples += N

            save_checkpoint(n_samples, (epoch, key, i, loss), model, optimizer)
            #print('validation loss:', validation_loss(model, criterion, val_data))

if __name__ == '__main__':
    train()
