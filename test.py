#!/usr/bin/env python

from train import load_checkpoint
from model import Img2Tex as Model
from data import load_images, load_vocab

import matplotlib.pyplot as plt
import torch

N = 4
def metric():
    return 0

def get_prediction(y, rev_vocab):
    return ' '.join([rev_vocab[i] for i in y])

def test():
    model = Model().cuda()
    test_data = load_images(['test'])[0]
    rev_vocab = load_vocab(rev=True)

    load_checkpoint(36724, model)

    with torch.no_grad():
        for key, batch in test_data.items():
            for i in range(0, len(batch[0]), N):
                x = batch[0][i:i+N].cuda()
                y = batch[1][i:i+N].cuda()
                y_pred = model(x)
                for j in range(0, N):
                    img = x[j].reshape(x[j].shape[1:]).cpu().numpy()
                    real = get_prediction(y[j], rev_vocab)
                    gen = get_prediction(y_pred[j].argmax(dim=0), rev_vocab)
                    print('real:', real)
                    print('gen:', gen)
                    plt.imshow(img)
                    plt.show()
                    exit(1)
        loss /= n
    return loss

    pass

if __name__ == '__main__':
    test()


