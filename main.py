from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import os.path as osp
import pdb
import pytz

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from datetime import datetime
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import pickle

from loader import Data_loader
from model import Model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance) 
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# TODO: NEED TO REWRITE to go through 15 choices
# TODO: NEED special data_loader mods for test
def test(args, tb_writer):
    # Some preparation
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, don\'t do this.')

    print ('Loading data')
    loader = Data_loader(15, args.emb, train=False)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d' % (loader.vocab_size, args.emb, loader.K, loader.feat_dim, args.hid))

    model = Model(vocab_size=loader.vocab_size,
                  emb_dim=args.emb,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=2,
                  pretrained_wemb=loader.pretrained_wemb,
                  symstream=args.sym,
                  noatt=args.noatt)

    model = model.cuda()

    if args.modelpath and os.path.isfile(args.modelpath):
        print ('Resuming from checkpoint %s' % (args.modelpath))
        ckpt = torch.load(args.modelpath)
        model.load_state_dict(ckpt['state_dict'])
    else:
        print(args.modelpath)
        raise SystemExit('Need to provide model path.')

    result = []
    for step in range(loader.n_batches):
        # Batch preparation
        q_batch, i_batch, s_batch, label_batch, img_indices = loader.next_batch()
    # result = []

    val_dict = load_cache_obj('val_dict')
    num_right = 0

    for i, ad_data in val_dict.items():
        
        query_batch = []
        img_feat_batch = []
        symbol_feat_batch = []
        label_batch = []

        img_feat = ad_data['img_feat']
        symbol_feat = ad_data['symbol_feat']

        for j in range(15):
            # question batch
            q = ad_data['query'][j]
            query_batch.append(q)

            # image batch
            img_feat_batch.append(img_feat) 

            # symbol batch
            symbol_feat_batch.append(symbol_feat) 

            # label batch
            label = ad_data['score'][j]
            label_batch.append(label)

        q_batch = np.asarray(query_batch)   # (batch, seqlen)
        i_batch = np.asarray(img_feat_batch)   # (batch, K, feat_dim)
        s_batch = np.asarray(symbol_feat_batch)   # (batch, K, feat_dim)
        label_batch = np.asarray(label_batch)

        # Cast to Variable
        q_batch = Variable(torch.from_numpy(q_batch))
        i_batch = Variable(torch.from_numpy(i_batch))
        s_batch = Variable(torch.from_numpy(s_batch))
        label_batch = Variable(torch.from_numpy(label_batch))
        q_batch, i_batch, s_batch, label_batch = q_batch.cuda(), i_batch.cuda(), s_batch.cuda(), label_batch.cuda()

        # Do one model forward and optimize
        # import pdb; pdb.set_trace()

        output = model(q_batch, i_batch, s_batch)
        # loss = loss_func(output, label_batch)

        # Calculate accuracy and loss
        confidence, pred_label = output.data.max(1)

        max_conf, max_conf_idx = torch.max(confidence, 0)

        pos_idx_pred = max_conf_idx.cpu().numpy()[0]
        pos_idx_true =  np.where(np.array(ad_data['score']) == 1)[0]

        if pos_idx_pred in pos_idx_true:
            num_right += 1

    # import pdb; pdb.set_trace()

    acc = num_right / len(val_dict.keys())
    print("num right / total: %d / %d" % (num_right, len(val_dict.keys())))
    print("accuracy: %s" % acc)

        # TODO: find row with max score and take the prediction
        # import pdb; pdb.set_trace()

        # # TODO: fix this
        # _, ix = output.data.max(1)
        # for i, qid in enumerate(a_batch):
        #     result.append({
        #         'question_id': qid,
        #         'answer': loader.a_itow[ix[i]]
        #     })

    # json.dump(result, open('result.json', 'w'))
    print ('Validation done')

def train(args, tb_writer):
    # Some preparation
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, don\'t do this.')

    print ('Loading data')
    loader = Data_loader(args.bsize, args.emb)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d' % (loader.vocab_size, args.emb, loader.K, loader.feat_dim,
                args.hid))
    print ('Initializing model')

    model = Model(vocab_size=loader.vocab_size,
                  emb_dim=args.emb,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=2,
                  pretrained_wemb=loader.pretrained_wemb,
                  symstream=args.sym,
                  noatt=args.noatt)

    model.apply(weights_init)

    # loss_func = nn.BCELoss()
    # weight = torch.Tensor([3./9., 1])
    weight = torch.Tensor([1., 1.])
    loss_func = nn.CrossEntropyLoss(weight=weight)
    
    # Move it to GPU
    model = model.cuda()
    loss_func = loss_func.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Continue training from saved model
    if args.modelpath and os.path.isfile(args.modelpath):
        print ('Resuming from checkpoint %s' % (args.modelpath))
        ckpt = torch.load(args.modelpath)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Training script
    print ('Start training.')
    for ep in range(args.ep):
        ep_loss = 0
        ep_correct = 0
        ep_zeros = 0.
        ep_total = 0.
        all_preds = {}
        for step in tqdm.tqdm(range(loader.n_batches)):
            # Batch preparation
            q_batch, i_batch, s_batch, label_batch, img_indices = loader.next_batch()
            # import pdb; pdb.set_trace()
            q_batch = Variable(torch.from_numpy(q_batch))
            i_batch = Variable(torch.from_numpy(i_batch))
            s_batch = Variable(torch.from_numpy(s_batch))
            label_batch = Variable(torch.from_numpy(label_batch))
            q_batch, i_batch, s_batch, label_batch = q_batch.cuda(), i_batch.cuda(), s_batch.cuda(), label_batch.cuda()

            # Do model forward
            output = model(q_batch, i_batch, s_batch)
            # logits = output[:, 1]
            # loss = loss_func(logits.squeeze(), label_batch.float())
            loss = loss_func(output, label_batch)
            # Calculate accuracy and loss
            # import pdb; pdb.set_trace()
            _, oix = output.data.max(1)
            aix = label_batch.data
            correct = torch.eq(oix, aix).sum()
            ep_correct += correct
            ep_loss += loss.data[0]
            zeros = (oix == 0).long().sum()
            ep_zeros += zeros
            ep_total += oix.numel()
            smop = F.softmax(output).data.cpu().numpy()
            for v, ix in enumerate(img_indices):
                if ix not in all_preds:
                    all_preds[ix] = {'p': [], 'gt': []}
                all_preds[ix]['gt'].append(label_batch[v].data.cpu().numpy())
                all_preds[ix]['p'].append(smop[v, 1])
            if step % 2 == 0 and step > 0:
                tqdm.tqdm.write('Epoch %02d(%03d/%03d), loss: %.3f, correct: %3d / %d (%.2f%%), zeros: %.3f%%' %
                        (ep+1, step, loader.n_batches, loss.data[0], correct, args.bsize, correct * 100 / args.bsize,
                            ep_zeros * 100. / ep_total))
                ep_zeros = 0.
                ep_total = 0.

            # write accuracy and loss to tensorboard
            total_batch_count = ep *  loader.n_batches + step
            acc_perc = correct / args.bsize
            tb_writer.add_scalar('train/loss', loss.data[0], total_batch_count)
            tb_writer.add_scalar('train/acc', acc_perc, total_batch_count)

            # compute gradient and do optim step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("")
        total_right = 0.
        total = 0.
        for ix, dat in all_preds.items():
            _preds = dat['p']
            _gt = dat['gt']
            if np.array(_gt).sum() == 0:
                continue
            maxpred = np.argmax(_preds)
            valat = _gt[maxpred]
            total_right += (valat > 0)[0]
            total += 1.
        print("Accuracy: {:%} ({} {})".format(total_right / total, float(total_right), float(total)))
        tb_writer.add_scalar('train/perc', float(total_right / total), ep)
            
        all_preds = {}
        # Save model after every epoch
        tbs = {
            'epoch': ep + 1,
            'loss': ep_loss / loader.n_batches,
            'accuracy': ep_correct * 100 / (loader.n_batches * args.bsize), 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        savebase = 'save/model-sym-'
        if args.sym:
            savebase += 'sym-'
        if args.noatt:
            savebase += 'noatt-'
        torch.save(tbs, savebase + str(ep+1) + '.pth.tar')
        # torch.save(tbs, 'save/model-sym-' + str(ep+1) + '.pth.tar')
        print ('Epoch %02d done, average loss: %.3f, average accuracy: %.2f%%' % (ep+1, ep_loss / loader.n_batches, ep_correct * 100 / (loader.n_batches * args.bsize)))


def load_cache_obj(name ):
    with open('data/ads/cache/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_tb_path(tb_dir, name):
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    run_name = ''
    if name:
        run_name += run_name
    now = datetime.now(pytz.timezone('US/Eastern'))
    run_name += '_%s' % now.strftime('%Y%m%d-%H%M%S')
    return osp.join(tb_dir, run_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Winner of VQA 2.0 in CVPR\'17 Workshop')
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--eval', action='store_true', help='set this to evaluate.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--ep', metavar='', type=int, default=50, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=512, help='batch size.')
    parser.add_argument('--hid', metavar='', type=int, default=512, help='hidden dimension.')
    parser.add_argument('--emb', metavar='', type=int, default=300, help='embedding dimension. (50, 100, 200, *300)')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained model path.')
    parser.add_argument('--name', metavar='', type=str, default=None, help='name of tb run')
    parser.add_argument('--tb_dir', metavar='', type=str, default='data/ads/tb', help='path to tb directory')
    parser.add_argument('--sym', metavar='', type=bool, default=False, help='symbolic stream toggle')
    parser.add_argument('--gpu', metavar='', type=int, default=0, help='gpu number')
    parser.add_argument('--noatt', metavar='', type=bool, default=False, help='Removes the attention component of the model')


    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    tb_path = get_tb_path(args.tb_dir, args.name)
    tb_writer = SummaryWriter(tb_path)

    # tb_writer.add_text('main/test', 'Hello World')

    if args.train:
        train(args, tb_writer)
    if args.eval:
        test(args, tb_writer)
    if not args.train and not args.eval:
        parser.print_help()
