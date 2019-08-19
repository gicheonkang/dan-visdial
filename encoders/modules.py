"""
Dual Attention Networks for Visual Reference Resolution in Visual Dialog
Gi-Cheon Kang, Jaeseo Lim, Byoung-Tak Zhang
https://arxiv.org/abs/1902.09368
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm
from .submodules import MultiHeadAttention, PositionwiseFeedForward
from .fc import FCNet

class REFER(nn.Module):
    """ This code is modified from Yu-Hsiang Huang's repository
        https://github.com/jadore801120/attention-is-all-you-need-pytorch
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.2):
        super(REFER, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q, m):
        enc_output, enc_slf_attn = self.slf_attn(q, m, m)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class FIND(nn.Module):
    """ This code is modified from Hengyuan Hu's repository.
        https://github.com/hengyuan-hu/bottom-up-attention-vqa
    """
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(FIND, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, v, 2048]
        q: [10, batch, 1024]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) 
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits
