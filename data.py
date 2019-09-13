import torch

import numpy as np
import scipy.misc
import os
import pickle

from model import rnn_max_steps, tex_token_size, device
data_dir = '/home/talnikar/projects/scratch/img2tex/data/'
#data_dir = '/home/talnikar/projects/im2markup/data/sample_bak/'

def load_vocab(rev=False):
    with open(data_dir + 'latex_vocab.txt') as f:
        vocab = [x.rstrip('\n') for x in f.readlines()]
        vocab.append('\SOS')
        vocab.append(' ')
        vocab.append('\EOS')
        if not rev:
            vocab = {x: i for i, x in enumerate(vocab)}
    return vocab

def get_prediction(y):
    if not hasattr(get_prediction, 'rev_vocab'):
        get_prediction.rev_vocab = load_vocab(rev=True)
    return ' '.join([get_prediction.rev_vocab[i] for i in y])

def load_images(sources):
    vocab = load_vocab()
    with open(data_dir + 'formulas.norm.lst') as f:
        tokens = [x.rstrip('\n').split(' ') for x in f.readlines()]
    data_list = []
    for src in sources:
        data_pkl = data_dir + src + '_data.pkl'
        if os.path.exists(data_pkl):
            with open(data_pkl, 'rb') as f:
                data_dict = pickle.load(f)
        else:
            data_dict = {}

            data_file = src + '_filter.lst'
            with open(data_dir + data_file) as f:
                data_from_file = [x.rstrip('\n').split(' ') for x in f.readlines()]
            for f_img, idx in data_from_file:
            #for f_img, idx in data_from_file[:1000]:
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
                if len(formula) > rnn_max_steps - 1:
                    continue
                try:
                    formula_idx = list(filter(lambda y: y != vocab[' '], [vocab[x] for x in formula]))
                except KeyError:
                    continue
                assert len(formula_idx) > 0
                formula_idx = np.concatenate((formula_idx, [vocab['\EOS']]))
                key = img.shape
                if key not in data_dict.keys():
                    data_dict[key] = []
                data_dict[key].append((img, formula_idx))

            for key in data_dict.keys():
                val = data_dict[key]
                images = np.array([x[0] for x in val]).astype(np.float32)
                formulas = [x[1] for x in val]
                #formulas = np.array([np.pad(x[1], (0, max_tokens-len(x[1])), 'constant', constant_values=whitespace) for x in val])#.astype(np.int32)
                
                data_dict[key] = (images, formulas)

        with open(data_pkl, 'wb') as f:
            pickle.dump(data_dict, f)
        data_list.append(data_dict)

    mem = 0
    for data_dict in data_list:
        for key in data_dict.keys():
            x, y = data_dict[key]
            x = torch.tensor(x[:,None,:,:], device=device)
            #y = torch.tensor(y, device=device)
            y = [torch.tensor(tmp, device=device).reshape(1, -1) for tmp in y]
            #x = torch.tensor(x[:,None,:,:])
            #y = torch.tensor(y)
            #mem += (np.prod(x.shape) + np.prod(y.shape))*4
            data_dict[key] = (x, y)
    print('data mem usage (GB):', mem/1024**3)

    return data_list
        
