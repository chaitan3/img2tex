#!/usr/bin/env python

import torch
import time

from model import Img2Tex as Model
from model import device
from data import load_images, get_prediction

#N = 4
N = 1
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
        teacher_loss = 0.
        vanilla_loss = 0.
        n = 0.
        for key, batch in data.items():
            for i in range(0, len(batch[0]), N):
                x = batch[0][i:i+N].cuda()
                y = batch[1][i:i+N][0].cuda()
                n += 1
                y_pred = model(x, y)
                real = get_prediction(y[0])
                gen = get_prediction(y_pred[0].argmax(dim=0))
                print(real)
                print(gen)
                teacher_loss += criterion(y_pred, y)
                #vanilla_loss += criterion(model(x), y)
                print('val {} {}/{}, losses: {}, {}'.format(key, i, len(batch[0]), teacher_loss.item()/n, vanilla_loss/n))
        loss /= n
    return vanilla_loss

def load_checkpoint(n_samples, model, optimizer=None):
    checkpoint = torch.load('checkpoint_{}.pth'.format(n_samples))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['data']

def train():
    model = Model().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1**(epoch+1))

    #start_epoch, start_key, _, _ = load_checkpoint('latest', model, optimizer)
    start_epoch, start_key = 0, None
    train_data, val_data = load_images(['train', 'validate'])

    data_keys = list(train_data.keys())
    if start_key:
        start_key = data_keys.index(start_key) + 1
    else:
        start_key = 0

    print('model parameter size (GB):', model_size(model)/1024**3)
    #print(validation_loss(model, criterion, val_data))
    #exit(1)
    #import pdb;pdb.set_trace()

    n_samples = 0
    for epoch in range(start_epoch, n_epochs):
        print('starting epoch', epoch)
        #scheduler.step()
        for key, batch in train_data.items():
            if data_keys.index(key) < start_key:
                continue
            batch_size = len(batch[0])
            #print(key, batch[1].shape)
            #continue
            for i in range(0, batch_size, N):
                start = time.time()
                x = batch[0][i:i+N].cuda()
                y = batch[1][i:i+N][0].cuda()
                #x = batch[0][i:i+N]
                #y = batch[1][i:i+N]

                optimizer.zero_grad()
                y_pred = model(x, y)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                
                total_norm = 0.
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    print('grad_norm', param_norm)
                total_norm = total_norm ** (1. / 2)
                print('total_norm', total_norm)

                end = time.time()

                print('epoch: {} {} {}/{}, loss: {}'.format(epoch, key, i, batch_size, loss.item()))
                print('time step:', end-start)
                n_samples += N

        save_checkpoint(n_samples, (epoch, key, i, loss), model, optimizer)
        #print('validation loss:', validation_loss(model, criterion, val_data))

if __name__ == '__main__':
    train()
