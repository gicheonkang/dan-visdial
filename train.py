"""
This code is modified from batra-mlp-lab's repository.
https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
"""
import argparse
import datetime
import gc
import math
import os
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataloader import VisDialDataset
from encoders import Encoder, DAN
from decoders import Decoder
from utils import utils

parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
DAN.add_cmdline_args(parser)

parser.add_argument_group('Encoder Decoder choice arguments')
parser.add_argument('-encoder', default='dan', help='encoder to use for training')
parser.add_argument('-decoder', default='disc', help='Decoder to use for training')
parser.add_argument_group('Optimization related arguments')
parser.add_argument('-num_epochs', default=12, type=int, help='Epochs')
parser.add_argument('-batch_size', default=80, type=int, help='Batch size')
parser.add_argument('-weight_decay', default=1e-5, type=float)
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument_group('Checkpointing related arguments')
parser.add_argument('-load_path', default='', help='Checkpoint to load path from')
parser.add_argument('-save_path', default='checkpoints/', help='Path to save checkpoints')
parser.add_argument('-save_step', default=1, type=int,
                        help='Save checkpoint after every save_step epochs')

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
if args.save_path == 'checkpoints/':
    args.save_path += start_time

np.random.seed(5912)
torch.cuda.manual_seed_all(5912)
torch.backends.cudnn.benchmark = True

# transfer all options to model
model_args = args

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------
for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------
dataset = VisDialDataset(args, ['train'])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=dataset.collate_fn)

# ----------------------------------------------------------------------------
# setting model args
# ----------------------------------------------------------------------------
# transfer some useful args from dataloader to model
for key in {'num_data_points', 'vocab_size', 'max_ques_count',
            'max_ques_len', 'max_ans_len'}:
    setattr(model_args, key, getattr(dataset, key))

# iterations per epoch
setattr(args, 'iter_per_epoch', 
    math.floor(dataset.num_data_points['train'] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------
encoder = Encoder(model_args)
decoder = Decoder(model_args, encoder)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                        lr=args.lr, weight_decay=args.weight_decay)
encoder.word_embed.init_embedding('data/glove/glove6b_init_300d_1.0.npy')

start_epoch = 0
if args.load_path != '':
    components = torch.load(args.load_path)
    encoder.load_state_dict(components.get('encoder', components))
    decoder.load_state_dict(components.get('decoder', components))
    optimizer.load_state_dict(components.get('optimizer', components))
    start_epoch = components['epoch']
    print("Loaded model from {}".format(args.load_path))
print("Decoder: {}".format(args.decoder))

args_for_save = encoder.args
encoder = nn.DataParallel(encoder).cuda()
decoder = nn.DataParallel(decoder).cuda()
criterion = criterion.cuda()


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
encoder.train()
decoder.train()
os.makedirs(args.save_path, exist_ok=True)
logger = utils.Logger(os.path.join(args.save_path, 'log.txt'))

running_loss = 0.0
train_begin = datetime.datetime.utcnow()
print("Training start time: {}".format(
    datetime.datetime.strftime(train_begin, '%d-%b-%Y-%H:%M:%S')))

for epoch in range(1, model_args.num_epochs + 1):
    if epoch == 1:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.9*1e-3, 1.0*1e-3)
    elif epoch == 2:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.9*1e-3, 0.8*1e-3)
    elif epoch == 3:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.7*1e-3, 0.8*1e-3)
    elif epoch == 4:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.7*1e-3, 0.6*1e-3)
    elif epoch == 5:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.5*1e-3, 0.6*1e-3)
    elif epoch == 6:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.5*1e-3, 0.4*1e-3)
    elif epoch == 7:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.3*1e-3, 0.4*1e-3)
    elif epoch == 8:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.3*1e-3, 0.15*1e-3)
    elif epoch == 9:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.075*1e-3, 0.15*1e-3)
    elif epoch == 10:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.075*1e-3, 0.0375*1e-3)
    elif epoch == 11:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.01875*1e-3, 0.0375*1e-3)
    elif epoch == 12:
        clr = utils.cyclic_lr(args.iter_per_epoch, 0.01875*1e-3, 0.009375*1e-3)

    for i, batch in enumerate(dataloader):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = Variable(batch[key])
                batch[key] = batch[key].cuda()
        # when trainset is odd number, it can be error with multi-GPU environment
        if i >= args.iter_per_epoch:
            continue

        # --------------------------------------------------------------------
        # forward-backward pass and optimizer step
        # --------------------------------------------------------------------
        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)
        cur_loss = criterion(dec_out, batch['ans_ind'].view(-1))
        optimizer.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()
        cur_loss.backward()
        nn.utils.clip_grad_norm(encoder.parameters(), 0.5)
        nn.utils.clip_grad_norm(decoder.parameters(), 0.5)
        optimizer.step()
        gc.collect()
        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * cur_loss.data[0]
        else:
            running_loss = cur_loss.data[0]

        if optimizer.param_groups[0]['lr'] > 5e-6:
            optimizer.param_groups[0]['lr'] = clr.lr(epoch, i)
        # --------------------------------------------------------------------
        # print after every few iterations
        # --------------------------------------------------------------------
        if i % 100 == 0:
            # print current time, running average, learning rate, iteration, epoch
            logger.write("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                datetime.datetime.utcnow() - train_begin, epoch,
                    (epoch - 1) * args.iter_per_epoch + i, running_loss,
                    optimizer.param_groups[0]['lr']))

    # ------------------------------------------------------------------------
    # save checkpoints and final model
    # ------------------------------------------------------------------------
    if epoch % args.save_step == 0:
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': args_for_save,
            'epoch': epoch
        }, os.path.join(args.save_path, 'dan_disc_epoch_{}.pth'.format(epoch)))        
torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': args_for_save,
    'epoch': args.num_epochs 
}, os.path.join(args.save_path, 'dan_disc_final.pth'))
