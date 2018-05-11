from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path as osp
import pickle

import numpy as np

class Data_loader:
    def __init__(self, batch_size=512, emb_dim=300, train=True):
        self.bsize = batch_size
        self.emb_dim = emb_dim # fix at 300 using caching
        self.train = train
        self.seqlen = 20    # hard-coded

        # with open('/u/rkdoshi/AdsVQA/data/ads/ad_data/ImageSets/train.txt', 'r') as f:
        #     train_img_fnames = f.readlines()
        #     train_img_fnames = [x.strip() for x in train_img_fnames]
        #     train_img_fnames = set([x.split('.')[0] for x in train_img_fnames]) # remove filetype

        # if train:
        #     with open('data/ads/ad_data/train/QA_Combined_Action_Reason_train.json', 'r') as f:
        #         text_annotations = json.load(f)
        # else: # load test stuff
        #     with open('data/ads/ad_data/train/QA_Combined_Action_Reason_test.json', 'r') as f:
        #         text_annotations = json.load(f)

        # {q_id: {'query': <Action-Reason string, 'img_id':<img_id>, 'img_feat': <np.array>,'score': <0/1>}}
        self.mem = load_cache_obj('train_dict')
        # mem_counter = 0

        # for img_fname, annotations in text_annotations.items():
        #     img_fname_prefix = img_fname.split('.')[0]

        #     if img_fname_prefix in train_img_fnames:

        #         pos = annotations[0]
        #         neg = [i for i in annotations[1] if i not in pos]

        #         img_feat_fname =  img_fname_prefix + '.npz'
        #         img_filepath = osp.join('/opt/visualai/ads/img_features/', img_feat_fname)
        #         img_feat = np.load(img_filepath)['feat'][()]
        #         img_feat = np.concatenate([v for v in img_feat.values()], axis=0)

        #         symbol_filepath = osp.join('/opt/visualai/ads/symbol_features/', img_feat_fname)
        #         symbol_feat = np.load(symbol_filepath)['feat'][()]
        #         symbol_feat = np.concatenate([v for v in symbol_feat.values()], axis=0)

        #         for annotation in pos:
        #             self.mem[mem_counter] = {
        #                 'query': annotation,
        #                 'img_fname_prefix': img_fname_prefix, # TODO: used?
        #                 'img_feat': img_feat,
        #                 'symbol_feat': symbol_feat,
        #                 'score': 1
        #             }
        #             mem_counter += 1

        #         for annotation in neg:
        #             self.mem[mem_counter] = {
        #                 'query': annotation,
        #                 'img_fname_prefix': img_fname_prefix, # TODO: used?
        #                 'img_feat': img_feat,
        #                 'symbol_feat': symbol_feat,
        #                 'score': 0,
        #             }
        #             mem_counter += 1

        self.n_queries = len(self.mem.keys())

        print ('Loading done')

        # initialize loader
        self.n_batches = self.n_queries // self.bsize
        self.K = 100 # hard-coded for now
        self.feat_dim = 1024  # hard-coded for now
        # self.init_pretrained_wemb(emb_dim)
        self.pretrained_wemb = load_cache_obj('glove_300d_pretrained_wemb')
        self.itow = load_cache_obj('glove_300d_itow')
        self.wtoi = load_cache_obj('glove_300d_wtoi')
        self.vocab_size = 400000 # hard-coded for now
        self.epoch_reset()

    # def init_pretrained_wemb(self, emb_dim):
    #     """From blog.keras.io"""
    #     embeddings_lookup = {}
    #     f = open('data/ads/glove/glove.6B.' + str(emb_dim) + 'd.txt')
    #     for line in f:
    #         values = line.split()
    #         word = values[0]
    #         embedding_vec = np.asarray(values[1:], dtype=np.float32)
    #         embeddings_lookup[word] = embedding_vec
    #     f.close()

    #     self.itow, self.wtoi = {}, {}
    #     self.vocab_size = len(embeddings_lookup.keys())
    #     self.pretrained_wemb = np.zeros((self.vocab_size, emb_dim))    
    #     for i, word in enumerate(embeddings_lookup.keys()):
    #         embedding_v = embeddings_lookup.get(word)
    #         if embedding_v is not None:
    #             self.pretrained_wemb[i, :] = embedding_v
    #         self.itow[i] = word
    #         self.wtoi[word] = i

    def epoch_reset(self):
        self.batch_ptr = 0
        np.random.shuffle(self.mem)


    def next_batch(self):
        """Return 3 things:
        query_batch -> (batch, seqlen)
        image feature -> (batch, K, feat_dim)
        label -> (batch); 0 or 1 scalar
        """
        if self.batch_ptr + self.bsize >= self.n_queries:
            self.epoch_reset()

        query_batch = []
        img_feat_batch = []
        symbol_feat_batch = []
        label_batch = []

        for b in range(self.bsize):
            # question batch
            q = [0] * self.seqlen
            for i, w in enumerate(self.mem[self.batch_ptr + b]['query']):
                if i >= self.seqlen:
                    break

                try:
                    q[i] = self.wtoi[w]
                except:
                    q[i] = 0    # validation questions may contain unseen word
            query_batch.append(q)

            # image batch
            img_feat = self.mem[self.batch_ptr + b]['img_feat']
            img_feat_batch.append(img_feat) 

            # symbol batch
            symbol_feat = self.mem[self.batch_ptr + b]['symbol_feat']
            symbol_feat_batch.append(symbol_feat) 

            # label batch
            label = self.mem[self.batch_ptr + b]['score']
            label_batch.append(label)

        self.batch_ptr += self.bsize
        query_batch = np.asarray(query_batch)   # (batch, seqlen)
        img_feat_batch = np.asarray(img_feat_batch)   # (batch, K, feat_dim)
        symbol_feat_batch = np.asarray(symbol_feat_batch)   # (batch, K, feat_dim)
        label_batch = np.asarray(label_batch)

        return query_batch, img_feat_batch, symbol_feat_batch, label_batch

def load_cache_obj(name):
    with open('data/ads/cache/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)