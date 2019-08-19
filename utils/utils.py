from __future__ import print_function
import os
import sys
import json
import functools
import operator
import glob
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle as cPickle

def create_glove_embedding_init(idx2word, glove_file):
    """
    Bilinear Attention Networks
    Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
    https://github.com/jnhwkim/ban-vqa
    """
    word2emb = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)

    for idx in range(1, len(idx2word)):
        word = ind2word.get(str(idx))
        if word not in word2emb:
            continue
        weights[idx-1] = word2emb[word]
    return weights

class Logger(object):
    """
    Bilinear Attention Networks
    Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
    https://github.com/jnhwkim/ban-vqa
    """
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

class cyclic_lr():
    """
    code for cyclir learning rate
    """
    def __init__(self, iter_per_epoch, base_lr, max_lr, epochs_per_cycle = 2):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.epochs_per_cycle = epochs_per_cycle
        self.iterations_per_epoch = iter_per_epoch
        self.step_size = (self.epochs_per_cycle*self.iterations_per_epoch)/2

    def iteration(self, epoch, batch_idx):
        return epoch*self.iterations_per_epoch + batch_idx

    def lr(self, epoch, batch_idx):
        cycle = np.floor(1+self.iteration(epoch, batch_idx)/(2*self.step_size))
        x = np.abs(self.iteration(epoch, batch_idx)/self.step_size - 2*cycle + 1)
        lr = self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))
        return lr

def load_imageid(img_root):
    img_ids = set()
    for image_path in glob.iglob(os.path.join(img_root, '*.jpg')):
        img_ids.add(int(image_path[-12:-4]))

    print('number of images: ', len(img_ids))
    assert len(img_ids) != 0
    return img_ids

if __name__ == "__main__":
    params = json.load(open('../data/visdial_1.0_params.json', 'r'))
    ind2word = params['ind2word']
    weights = create_glove_embedding_init(ind2word, '../data/glove/glove.6B.300d.txt')
    np.save('../data/glove/glove6b_init_300d_1.0.npy', weights)

