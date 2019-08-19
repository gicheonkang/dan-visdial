"""
Dual Attention Networks for Visual Reference Resolution in Visual Dialog
Gi-Cheon Kang, Jaeseo Lim, Byoung-Tak Zhang
https://arxiv.org/abs/1902.09368
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from .fc import FCNet
from .modules import REFER, FIND
from utils import WordEmbedding, DynamicRNN

class DAN(nn.Module):
    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Encoder specific arguments')
        parser.add_argument('-embed_size', default=300,
                                help='Size of the input word embedding')
        parser.add_argument('-hidden_size', default=512,
                                help='Size of the multimodal embedding')
        parser.add_argument('-dropout', default=0.5, help='Dropout')
        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bn = nn.BatchNorm1d(args.hidden_size)
        self.word_embed = WordEmbedding(args.vocab_size, 300, .0)
        self.sent_embed = nn.LSTM(args.embed_size, args.hidden_size, 2, dropout=args.dropout, batch_first=True)
        self.sent_embed = DynamicRNN(self.sent_embed)
        self.hist_embed = nn.LSTM(args.embed_size, args.hidden_size, 2, dropout=args.dropout, batch_first=True)
        self.hist_embed = DynamicRNN(self.hist_embed)
        self.bup_att = FIND(2048, 1024, 1024)

        self.q_net = FCNet([1024, 1024])
        self.v_net = FCNet([2048, 1024])
        self.linear = nn.Linear(args.hidden_size*2, args.hidden_size)

        self.layer_stack = nn.ModuleList([
            REFER(d_model=512, d_inner=1024, n_head=4, d_k=256, d_v=256, dropout=0.2)
            for _ in range(2)])

    def add_entry(self, mem, hist, hist_len):
        h_emb = self.word_embed(hist)
        h_emb = self.hist_embed(h_emb, hist_len).unsqueeze(1)

        if mem is None: mem = h_emb
        else: mem = torch.cat((mem, h_emb), 1)
        return mem
    
    def refer_module(self, mem, q):
        '''
        q : [b, 512]
        mem : [b, number of entry, 512]
        '''
        context = q.unsqueeze(1)
        for enc_layer in self.layer_stack:
            context, _ = enc_layer(context, mem)
        return context.squeeze(1)

    def find_module(self, v, l):
        att = self.bup_att(v, l)
        v_emb = (att * v).sum(1)
        q_repr = self.q_net(l)
        v_repr = self.v_net(v_emb)
        return q_repr * v_repr

    def forward(self, batch):
        v = batch['img_feat']
        q = batch['ques']
        ql = batch['ques_len']
        c = batch['cap']
        cl = batch['cap_len']
        h = batch['hist']
        hl = batch['hist_len']

        batch, num_dial, maxseq = q.size()        
        mem = self.add_entry(None, c, cl)
        enc_outs = [] 

        for i in range(num_dial):
            q_emb = self.word_embed(q[:, i, :])
            q_emb = self.sent_embed(q_emb, ql[:, i])

            hist = self.refer_module(mem, q_emb)
            ref_aware = torch.cat((q_emb, hist), 1)
            joint = self.find_module(v, ref_aware)
            enc_outs.append(joint)

            # write history embedding to memory
            if i != num_dial-1:
                mem = self.add_entry(mem, h[:, i+1, :], hl[:, i+1])    

        enc_out = torch.stack(enc_outs, 1)
        enc_out = self.linear(enc_out)
        enc_out = self.bn(enc_out.view(-1, self.args.hidden_size))
        enc_out = F.dropout(enc_out, 0.5)
        return enc_out
