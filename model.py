# pylint: disable-all
# flake8: noqa
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, vocab_size, emb_dim, K, feat_dim,
                 hid_dim, out_dim, pretrained_wemb, symstream=False,
                 noatt=False):
        super(Model, self).__init__()
        """
        Args:
            vocab_size: vocabularies to embed (question vocab size)
            emb_dim: GloVe pre-trained dimension -> 300
            K: image bottom-up attention locations -> 36
            feat_dim: image feature dimension -> 2048
            hid_dim: hidden dimension -> 512
            out_dim: multi-label regression output -> (answer vocab size)
            pretrained_wemb: as its name
        """
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.K = K
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.symstream = symstream

        # question encoding
        self.wembed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        
        # Opt for simpler, concat then predict (reduces feature dimensionality first)
        self.noatt = noatt

        # gated tanh activation
        if not noatt:
            self.gt_W_img_att = nn.Linear(feat_dim + hid_dim, hid_dim)
            self.gt_W_prime_img_att = nn.Linear(feat_dim + hid_dim, hid_dim)
        self.gt_W_question = nn.Linear(hid_dim, hid_dim)
        self.gt_W_prime_question = nn.Linear(hid_dim, hid_dim)
        self.gt_W_img = nn.Linear(feat_dim, hid_dim)
        self.gt_W_prime_img = nn.Linear(feat_dim, hid_dim)
        # self.gt_W_clf = nn.Linear(hid_dim, hid_dim)
        # self.gt_W_prime_clf = nn.Linear(hid_dim, hid_dim)

        # gated tanh for symbol stream (symstream)
        if symstream:
            if not noatt:
                self.gt_W_sy_att = nn.Linear(feat_dim + hid_dim, hid_dim)
                self.gt_W_prime_sy_att = nn.Linear(feat_dim + hid_dim, hid_dim)
            self.gt_W_sy = nn.Linear(feat_dim, hid_dim)
            self.gt_W_prime_sy = nn.Linear(feat_dim, hid_dim)
            self.gt_W_question_sy = nn.Linear(hid_dim, hid_dim)
            self.gt_W_prime_question_sy = nn.Linear(hid_dim, hid_dim)

        # image attention
        if not noatt:
            self.att_wa = nn.Linear(hid_dim, 1)

        # symbol attention
        if symstream:
            self.sy_att_wa = nn.Linear(hid_dim, 1)
            # output classifier
            self.clf_w = nn.Linear(hid_dim * 2, out_dim)
        else:
            self.clf_w = nn.Linear(hid_dim, out_dim)

        # initialize word embedding layer weight
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))

    def forward(self, question, image, symbol):
        """
        question -> shape (batch, seqlen)
        image -> shape (batch, K, feat_dim)
        """
        # question encoding
        emb = self.wembed(question)                 # (batch, seqlen, emb_dim)
        enc, hid = self.gru(emb.permute(1, 0, 2))   # (seqlen, batch, hid_dim)
        qenc = enc[-1]                              # (batch, hid_dim)

        image = F.normalize(image.float(), dim=-1)  # (batch, K, feat_dim)
        q_head = self._gated_tanh(
            qenc,
            self.gt_W_question,
            self.gt_W_prime_question)
        if self.symstream:
            symbol = F.normalize(symbol.float(), dim=-1)

        if self.noatt:
            image = image.mean(dim=1)
            v_head = self._gated_tanh(image, self.gt_W_img, self.gt_W_prime_img)
            # v_gat_head = torch.mul(v_head, q_head)

            if not self.symstream:
                concat_vq = torch.cat((v_head, q_head), dim=1)
                output = self.clf_w(concat_vq) 
                return output

            symbol = symbol.mean(dim=1)
            s_head = self._gated_tanh(symbol, self.gt_W_sy, self.gt_W_prime_sy)
            # s_gat_head = torch.mul(s_head, q_head)
            concat_vqs = torch.cat((v_head, q_head, s_head), dim=1)
            output = self.clf_w(concat_vqs)
            return output

        else:
            # image (v) encoding
            qenc_reshape = qenc.repeat(1, self.K).view(-1, self.K, self.hid_dim)
            # (batch, K, feat_dim + hid_dim)
            concated = torch.cat((image, qenc_reshape), dim=-1)
            concated = self._gated_tanh(
                concated,
                self.gt_W_img_att,
                self.gt_W_prime_img_att)   # (batch, K, hid_dim)

            a = self.att_wa(concated)                           # (batch, K, 1)
            a = F.softmax(a.squeeze())                          # (batch, K)
            v_head = torch.bmm(a.unsqueeze(1), image).squeeze()  # (batch, feat_dim)
            v_head = self._gated_tanh(v_head, self.gt_W_img, self.gt_W_prime_img)
            # v_gat_head = torch.mul(q_head, v_head)         # (batch, hid_dim)

            # Symbol attention
            if not self.symstream:
                concat_vq =  torch.cat((v_head, q_head), dim=1)
                output = self.clf_w(concat_vq)
                return output

            concated_sym = torch.cat((symbol, qenc_reshape), dim=-1)
            concated_sym = self._gated_tanh(
                concated_sym, 
                self.gt_W_sy_att, 
                self.gt_W_prime_sy_att)
            a_sy = self.sy_att_wa(concated_sym)
            a_sy = F.softmax(a.squeeze())
            s_head = torch.bmm(a_sy.unsqueeze(1), symbol).squeeze()
            s_head = self._gated_tanh(s_head, self.gt_W_sy, self.gt_W_prime_sy)
            # s_gat_head = torch.mul(q_head, s_head)

            # output classifier
            concat_vqs = torch.cat((v_head, q_head, s_head), dim=1)
            output = self.clf_w(concat_vqs)
            return output               # (batch, out_dim)

    def _gated_tanh(self, x, W, W_prime):
        """
        Implement the gated hyperbolic tangent non-linear activation
            x: input tensor of dimension m
        """
        y_tilde = F.tanh(W(x))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y
