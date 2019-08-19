"""
This code is modified from batra-mlp-lab's repository.
https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
"""
import torch
import torch.nn as nn
from utils import DynamicRNN, WordEmbedding

class DiscriminativeDecoder(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        # share word embedding
        self.word_embed = encoder.word_embed
        self.opt_embed = nn.LSTM(args.embed_size, 512, batch_first=True, dropout=args.dropout)
        self.opt_embed = DynamicRNN(self.opt_embed)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_out, batch):
        """Given encoder output `enc_out` and candidate output option sequences,
        predict a score for each output sequence.

        Arguments
        ---------
        enc_out : torch.autograd.Variable
            Output from the encoder through its forward pass. (b, rnn_hidden_size)
        options : torch.LongTensor
            Candidate answer option sequences. (b, num_options, max_len + 1) 
        """
        options = batch['opt']          #[b, num_dial, 100, 20]
        options_len = batch['opt_len']  #[b, num_dial, 100]
        # word embed options
        options = options.view(options.size(0) * options.size(1), options.size(2), -1) #[bx10, 100, 20]
        options_len = options_len.view(options_len.size(0) * options_len.size(1), -1)  #[bx10, 100]
        batch_size, num_options, max_opt_len = options.size()
        # score each option
        scores = []
        for opt_id in range(num_options):
            opt = options[:, opt_id, :]
            optl = options_len[:, opt_id]
            opt = self.word_embed(opt)
            o_emb = self.opt_embed(opt, optl)
            scores.append(torch.sum(o_emb * enc_out, 1))
            
        # return scores
        scores = torch.stack(scores, 1)
        log_probs = self.log_softmax(scores)
        return log_probs
