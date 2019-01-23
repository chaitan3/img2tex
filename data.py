import torch

import numpy as np
import scipy.misc
import os
import pickle

from model import rnn_max_steps, tex_token_size, device
data_dir = 'data/'

def load_vocab(rev=False):
    with open(data_dir + 'latex_vocab.txt') as f:
        vocab = [x.rstrip('\n') for x in f.readlines()]
        vocab.append(' ')
        if not rev:
            vocab = {x: i for i, x in enumerate(vocab)}
    return vocab


def load_images(sources):
    vocab = load_vocab()
    with open(data_dir + 'formulas.norm.lst') as f:
        tokens = [x.rstrip('\n').split(' ') for x in f.readlines()]
    data_list = []
    for src in sources:
        data_pkl = data_dir + src + '_data.pkl'
        if os.path.exists(data_pkl):
            with open(data_pkl, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}

            data_file = src + '_filter.lst'
            with open(data_dir + data_file) as f:
                data_list = [x.rstrip('\n').split(' ') for x in f.readlines()]
            for f_img, idx in data_list:
                print(data_file, idx, f_img)
                img = scipy.misc.imread(data_dir + 'images_processed/' + f_img, mode='L')
                # convert to floating point in range [-1, 1]
                img = (img-127.5)/127.5
                # downsampling and padding
                #img = np.pad(img, ((8,8), (8,8)), 'constant', constant_values=1)
                #img = img[::2, ::2]
                #h, w = img.shape
                #i = 0
                #print(h, w)
                #while w > buckets[i][0]:
                #    i += 1
                #while h > buckets[i][1]:
                #    i += 1
                #refw, refh = buckets[i]
                #img = np.pad(img, ((0,refh-h), (0,refw-w)), 'constant', constant_values=1)
                #assert img.shape == (refh, refw)

                formula = tokens[int(idx)]
                if len(formula) > rnn_max_steps:
                    continue
                try:
                    formula_idx = np.array([vocab[x] for x in formula])
                except KeyError:
                    continue
                key = img.shape
                if key not in data.keys():
                    data[key] = []
                data[key].append((img, formula_idx))

            for key in data.keys():
                val = data[key]
                images = np.array([x[0] for x in val]).astype(np.float32)
                # hack till input feeding is implemented
                #max_tokens = max([len(x[1]) for x in val])
                max_tokens = rnn_max_steps
                whitespace = vocab[' ']
                formulas = np.array([np.pad(x[1], (0, max_tokens-len(x[1])), 'constant', constant_values=whitespace) for x in val])#.astype(np.int32)
                
                data[key] = (images, formulas)

        with open(data_pkl, 'wb') as f:
            pickle.dump(data, f)
        data_list.append(data)

    mem = 0
    for data in data_list:
        for key in data.keys():
            x, y = data[key]
            #x = torch.tensor(x[:,None,:,:], device=device)
            #y = torch.tensor(y, device=device)
            x = torch.tensor(x[:,None,:,:])
            y = torch.tensor(y)
            mem += (np.prod(x.shape) + np.prod(y.shape))*4
            data[key] = (x, y)
    print('data mem usage (GB):', mem/1024**3)

    return data_list
        
