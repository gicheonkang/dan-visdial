"""
This code is modified from batra-mlp-lab's repository.
https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
"""
import os
import json
from six import iteritems

import h5py
import numpy as np
from tqdm import tqdm
import _pickle as cPickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import itertools
import utils

class VisDialDataset(Dataset):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Dataloader specific arguments')
        parser.add_argument('-input_img', default='data/{}_feature/{}_btmup_f.hdf5',
                                help='HDF5 file with image features')
        parser.add_argument('-input_img2idx', default='data/{}_feature/{}_imgid2idx.pkl',
                                help='HDF5 file with image features')
        parser.add_argument('-input_ques', default='data/visdial_1.0_data.h5',
                                help='HDF5 file with preprocessed questions')
        parser.add_argument('-input_json', default='data/visdial_1.0_params.json',
                                help='JSON file with image paths and vocab')
        return parser

    def __init__(self, args, subsets):
        """Initialize the dataset with splits given by 'subsets', where
        subsets is taken from ['train', 'val', 'test']
        """
        super().__init__()
        self.args = args
        self.subsets = tuple(subsets)

        print("Dataloader loading json file: {}".format(args.input_json))
        with open(args.input_json, 'r') as info_file:
            info = json.load(info_file)
            # possible keys: {'ind2word', 'word2ind', 'unique_img_(split)'}
            for key, value in iteritems(info):
                setattr(self, key, value)

        word_count = len(self.word2ind)
        self.vocab_size = word_count
        print("Vocab size : {}".format(self.vocab_size))

        # construct reverse of word2ind after adding tokens
        self.ind2word = {
            int(ind): word
            for word, ind in iteritems(self.word2ind)
        }

        print("Dataloader loading h5 file: {}".format(args.input_ques))
        ques_file = h5py.File(args.input_ques, 'r')

        args.input_img = args.input_img.format(self.subsets[0], self.subsets[0])
        print("Dataloader loading h5 file: {}".format(args.input_img))
        img_file = h5py.File(args.input_img, 'r')

        args.input_img2idx = args.input_img2idx.format(self.subsets[0], self.subsets[0]) 
        self.img_id2idx = cPickle.load(open(args.input_img2idx, 'rb'))
        # load all data mats from ques_file into this
        self.data = {}

        # map from load to save labels
        io_map = {
            'ques_{}': '{}_ques',
            'ques_length_{}': '{}_ques_len',
            'ans_{}': '{}_ans',
            'ans_length_{}': '{}_ans_len',
            'img_pos_{}': '{}_img_pos',
            'cap_{}': '{}_cap',
            'cap_length_{}': '{}_cap_len',
            'opt_{}': '{}_opt',
            'opt_length_{}': '{}_opt_len',
            'opt_list_{}': '{}_opt_list',
            'num_rounds_{}': '{}_num_rounds',
            'ans_index_{}': '{}_ans_ind'
        }

        # processing every split in subsets
        for dtype in subsets:  # dtype is in ['train', 'val', 'test']
            print("\nProcessing split [{}]...".format(dtype))
            # read the question, answer, option related information
            for load_label, save_label in iteritems(io_map):
                if load_label.format(dtype) not in ques_file:
                    continue
                self.data[save_label.format(dtype)] = torch.from_numpy(
                    np.array(ques_file[load_label.format(dtype)], dtype='int64'))

            # load the object detection (Faster-RCNN) feature
            print("Reading image features...")
            self.img_feats = torch.from_numpy(np.array(img_file.get('image_features')))
            self.spatials = torch.from_numpy(np.array(img_file.get('spatial_features')))
            self.pos_boxes = torch.from_numpy(np.array(img_file.get('pos_boxes')))
            
            print('img feat size: ', self.img_feats.size())
            # save image features
            self.data[dtype + '_img_fv'] = self.img_feats
            img_fnames = getattr(self, 'unique_img_' + dtype)
            self.data[dtype + '_img_fnames'] = img_fnames

            # record some stats, will be transferred to encoder/decoder later
            # assume similar stats across multiple data subsets
            # maximum number of questions per image, ideally 10
            self.max_ques_count = self.data[dtype + '_ques'].size(1)
            # maximum length of question
            self.max_ques_len = self.data[dtype + '_ques'].size(2)
            # maximum length of answer
            self.max_ans_len = self.data[dtype + '_ans'].size(2)

        self.num_data_points = {}
        self.num_data_points[dtype] = self.data[dtype + '_cap_len'].size(0)
        print("[{0}] no. of threads: {1}".format(dtype, self.num_data_points[dtype]))

        print("\tMax no. of rounds: {}".format(self.max_ques_count))
        print("\tMax ques len: {}".format(self.max_ques_len))
        print("\tMax ans len: {}".format(self.max_ans_len))

        # prepare history
        for dtype in subsets:
            self._process_history(dtype)
            # 1 indexed to 0 indexed
            self.data[dtype + '_opt'] -= 1
            if dtype + '_ans_ind' in self.data:
                self.data[dtype + '_ans_ind'] -= 1

        # default pytorch loader dtype is set to train
        if 'train' in subsets:
            self._split = 'train'
        else:
            self._split = subsets[0]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets  # ['train', 'val', 'test']
        self._split = split

    # ------------------------------------------------------------------------
    # methods to override - __len__ and __getitem__ methods
    # ------------------------------------------------------------------------
    def __len__(self):
        return self.num_data_points[self._split]

    def __getitem__(self, idx):
        dtype = self._split
        item = {'index': idx}
        item['num_rounds'] = self.data[dtype + '_num_rounds'][idx]

        # get image features
        item['img_fnames'] = self.data[dtype + '_img_fnames'][idx]
        feature_idx = self.img_id2idx[self.data[dtype + '_img_fnames'][idx]]
        item['img_feat'] = self.img_feats[self.pos_boxes[feature_idx][0]:self.pos_boxes[feature_idx][1],:]

        # get question tokens
        item['ques'] = self.data[dtype + '_ques'][idx]
        item['ques_len'] = self.data[dtype + '_ques_len'][idx]

        item['cap'] = self.data[dtype + '_cap'][idx]
        item['cap_len'] = self.data[dtype + '_cap_len'][idx]

        item['ans'] = self.data[dtype + '_ans'][idx]
        item['ans_len'] = self.data[dtype + '_ans_len'][idx]

        # get history tokens
        item['hist_len'] = self.data[dtype + '_hist_len'][idx]
        item['hist'] = self.data[dtype + '_hist'][idx]

        # get options tokens
        opt_inds = self.data[dtype + '_opt'][idx]
        opt_size = list(opt_inds.size())    
        new_size = torch.Size(opt_size + [-1])
        ind_vector = opt_inds.view(-1)

        option_in = self.data[dtype + '_opt_list'].index_select(0, ind_vector)
        option_in = option_in.view(new_size)

        opt_len = self.data[dtype + '_opt_len'].index_select(0, ind_vector)
        opt_len = opt_len.view(opt_size)

        item['opt'] = option_in
        item['opt_len'] = opt_len
        if dtype != 'test':
            ans_ind = self.data[dtype + '_ans_ind'][idx]
            item['ans_ind'] = ans_ind.view(-1)

        # convert zero length sequences to one length
        # this is for handling empty rounds of v1.0 test, they will be dropped anyway
        if dtype == 'test':
            item['ques_len'][item['ques_len'] == 0] += 1
            item['opt_len'][item['opt_len'] == 0] += 1
            item['hist_len'][item['hist_len'] == 0] += 1
        return item

    #-------------------------------------------------------------------------
    # collate function utilized by dataloader for batching
    #-------------------------------------------------------------------------
    def collate_fn(self, batch):
        dtype = self._split
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}
        for key in merged_batch:
            if key in {'index', 'num_rounds', 'img_fnames'}:
                out[key] = merged_batch[key]
            elif key in {'cap_len'}:
                out[key] = torch.Tensor(merged_batch[key]).long()
            elif key in {'img_feat'}:
                num_max_boxes = max([x.size(0) for x in merged_batch['img_feat']])
                out[key] = torch.stack([F.pad(x, (0,0,0,num_max_boxes-x.size(0))).data for x in merged_batch['img_feat']], 0)
            else:
                out[key] = torch.stack(merged_batch[key], 0)

        # Dynamic shaping of padded batch
        out['hist'] = out['hist'][:, :, :torch.max(out['hist_len'])].contiguous()
        out['ques'] = out['ques'][:, :, :torch.max(out['ques_len'])].contiguous()
        out['opt'] = out['opt'][:, :, :, :torch.max(out['opt_len'])].contiguous()
        out['cap'] = out['cap'][:, :torch.max(out['cap_len'])].contiguous()

        batch_keys = ['img_fnames', 'num_rounds', 'img_feat', 'ques', 'ques_len', 'opt', 'opt_len', 
                      'cap', 'cap_len', 'hist', 'hist_len']
        if dtype != 'test':
            batch_keys.append('ans_ind')
        return {key: out[key] for key in batch_keys}

    #-------------------------------------------------------------------------
    # preprocessing functions
    #-------------------------------------------------------------------------
    def _process_history(self, dtype):
        """Process caption as well as history. Optionally, concatenate history 
        for lf-encoder."""
        captions = self.data[dtype + '_cap']
        questions = self.data[dtype + '_ques']
        ques_len = self.data[dtype + '_ques_len']
        cap_len = self.data[dtype + '_cap_len']
        max_ques_len = questions.size(2)

        answers = self.data[dtype + '_ans']
        ans_len = self.data[dtype + '_ans_len']
        num_convs, num_rounds, max_ans_len = answers.size()

        history = torch.zeros(num_convs, num_rounds, max_ques_len + max_ans_len).long()
        hist_len = torch.zeros(num_convs, num_rounds).long()

        # go over each question and append it with answer
        for th_id in range(num_convs):
            clen = cap_len[th_id]
            hlen = min(clen, max_ques_len + max_ans_len)
            for round_id in range(num_rounds):
                if round_id == 0:
                    # first round has caption as history
                    history[th_id][round_id][:max_ques_len + max_ans_len] \
                        = captions[th_id][:max_ques_len + max_ans_len]
                else:
                    qlen = ques_len[th_id][round_id - 1]
                    alen = ans_len[th_id][round_id - 1]
                    # else, history is just previous round question-answer pair
                    if qlen > 0:
                        history[th_id][round_id][:qlen] = questions[th_id][round_id - 1][:qlen]
                    if alen > 0:
                        history[th_id][round_id][qlen:qlen + alen] \
                            = answers[th_id][round_id - 1][:alen]
                    hlen = alen + qlen
                # save the history length
                hist_len[th_id][round_id] = hlen

        self.data[dtype + '_hist'] = history
        self.data[dtype + '_hist_len'] = hist_len
