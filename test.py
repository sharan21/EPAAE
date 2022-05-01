import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from nltk.translate.bleu_score import sentence_bleu
from vocab import Vocab
from model import *
from utils import *
# from preproc import create_toy_dataset
from batchify import get_batches, get_batches_annotated
from nlgeval import compute_metrics
# from train import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', metavar='DIR', default=None,
                    help='Name of dataset for bert pretrained')
parser.add_argument('--checkpoint', metavar='DIR', default=None,
                    help='checkpoint directory')
parser.add_argument('--output', metavar='FILE',
                    help='output file name (in checkpoint directory)')
parser.add_argument('--data', metavar='FILE',
                    help='path to data file')
parser.add_argument('--enc', default='mu', metavar='M',
                    choices=['mu', 'z'],
                    help='encode to mean of q(z|x) or sample z from q(z|x)')
parser.add_argument('--dec', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='decoding algorithm')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--max-len', type=int, default=35, metavar='N',
                    help='max sequence length')
parser.add_argument('--dataset', default=None, metavar='M',
                    choices=['yelp', 'snli', 'dnli', 'qqp', 'scitail', 'ppr', 'voices', 'tenses'])
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate on data file')
parser.add_argument('--reconstruct', action='store_true',
                    help='reconstruct data file')
parser.add_argument('--sample', action='store_true',
                    help='sample sentences from prior')
parser.add_argument('--tsne', action='store_true',
                    help='plot tsne of lspace wrt labels')
parser.add_argument('--tst-on-test', action='store_true',
                    help='TST exp for a particular dataset')
parser.add_argument('--tst-on-test-finegrained', action='store_true',
                    help='TST exp for a particular dataset')
parser.add_argument('--toy-tsne', action='store_true',
                    help='plot tsne of lspace wrt labels for toy dataset')
parser.add_argument('--mean-flip-dist', action='store_true',
                    help='plot tsne of lspace wrt labels for toy dataset')
parser.add_argument('--arithmetic', action='store_true',
                    help='compute vector offset avg(b)-avg(a) and apply to c')
parser.add_argument('--interpolate', action='store_true',
                    help='interpolate between pairs of sentences')
parser.add_argument('--latent-nn', action='store_true',
                    help='find nearest neighbor of sentences in the latent space')
parser.add_argument('--m', type=int, default=100, metavar='N',
                    help='num of samples for importance sampling estimate')
parser.add_argument('--n', type=int, default=5, metavar='N',
                    help='num of sentences to generate for sample/interpolate')
parser.add_argument('--k', type=float, default=2, metavar='R',
                    help='k * offset for vector arithmetic')
parser.add_argument('--seed', type=int, default=1,
                    metavar='N', help='random seed')
parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
parser.add_argument('--output-size', type=int, default=2,
                    metavar='N', help='number of classes in dataset')
parser.add_argument('--gpu', type=int, default=0,
                    metavar='N', help='ID of gpu')
parser.add_argument('--model-name', metavar='FILE',
                    default='model.pt', help='name of model')


def get_model(path, vocab, is_classifier=False):
    print("loading model from path {}".format(path))
    ckpt = torch.load(path)
    train_args = ckpt['args']
    if(not is_classifier):
        model = {'dae': DAE, 'vae': VAE, 'aae': AAE, 'van': VanillaAE}[
            train_args.model_type](vocab, train_args).to(device)
    else:
        model = Classifier(vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model, train_args

def encode(sents):
    assert args.enc == 'mu' or args.enc == 'z'
    batches, order = get_batches(sents, vocab, args.batch_size, device)
    z = []
    for inputs, _ in batches:

        mu, logvar, _ = model.encode(inputs, train_args, is_train=True)
        if args.enc == 'mu':
            zi = mu
        else:
            zi = reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_

def encode_annotated(sents):
    assert args.enc == 'mu' or args.enc == 'z'
    batches, order = get_batches_annotated(sents, vocab, args.batch_size, device, cbow=None, num_classes=args.output_size)
    z, l = [], []
    for inputs, targets, _ in batches:
        mu, logvar, _ = model.encode(inputs, train_args, is_train=True)
        zi = mu if args.enc == 'mu' else reparameterize(mu, logvar)
        if(len(targets.shape) == 1):
            targets = targets.unsqueeze(dim=0)        
        z.extend(zi.detach().cpu().numpy())
        l.extend(targets.detach().cpu().numpy())
    z, l = np.array(z), np.array(l)
    #get back original order of z to match sents
    z[np.array(order)[:z.shape[0]]] = z
    l[np.array(order)[:z.shape[0]]] = l
    return z, l

def decode(z):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+args.batch_size], device=device)
        outputs = model.generate(zi, args.max_len, args.dec).t()
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
        i += args.batch_size
    return strip_eos(sents)

if __name__ == '__main__':
    args = parser.parse_args()

    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.gpu)) if cuda else 'cpu'

    if(args.checkpoint != None):
        vocab = Vocab(os.path.join(args.checkpoint, 'vocab.txt'))
        model, train_args = get_model(os.path.join(
            args.checkpoint, args.model_name), vocab)

    if args.evaluate:
        sents = load_sent(args.data)
        batches, _ = get_batches(sents, vocab, args.batch_size, device)
        meters = evaluate(model, batches)
        print(' '.join(['{} {:.2f},'.format(k, meter.avg)
                        for k, meter in meters.items()]))

    if args.tst_on_test:  # TST experiment on test.csv of dataset
        expd = args.checkpoint + '/exps-'+args.model_name
        if not os.path.exists(expd):
            os.mkdir(expd)
        base = './data/{}-annotated/'.format(args.dataset)
        n_wrong, tot = 0, 0
        for l in range(2):  # 2 labels
            # create test.(src)to(dst)
            src_lbl = str(l)
            dst_lbl = '1' if src_lbl == '0' else '0'
            fa, fb, fc = os.path.join(base, 'test.'+src_lbl), os.path.join(
                base, 'test.'+dst_lbl), os.path.join(base, 'test.'+src_lbl)
            print("performing TST on {}, src_lbl: {}, dst_lbl: {}".format(fc, fa, fb))
            sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
            za, zb, zc = encode(sa), encode(sb), encode(sc)
            zd = zc + args.k * (zb.mean(axis=0) - za.mean(axis=0))
            sd = decode(zd)
            output = os.path.join(expd, 'test.'+src_lbl+'to'+dst_lbl)
            write_sent(sd, output)
    
    if args.tst_on_test_finegrained:  # TST experiment on test.csv of dataset
        expd = args.checkpoint + '/exps-'+args.model_name
        if not os.path.exists(expd):
            os.mkdir(expd)
        base = './data/{}-annotated/'.format(args.dataset)
        n_wrong, tot = 0, 0
        for l in range(2):  # 2 labels
            # create test.(src)to(dst)
            src_lbl = str(l)
            dst_lbl = '1' if src_lbl == '0' else '0'
            fa, fb, fc = os.path.join(base, 'test.'+src_lbl), os.path.join(
                base, 'test.'+dst_lbl), os.path.join(base, 'test.'+src_lbl)
            print("performing TST on {}, src_lbl: {}, dst_lbl: {}".format(fc, fa, fb))
            sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
            za, zb, zc = encode(sa), encode(sb), encode(sc)
            zd = zc + args.k * (zb.mean(axis=0) - za.mean(axis=0))
            sd = decode(zd)
            output = os.path.join(expd, 'test.'+src_lbl+'to'+dst_lbl+'.{}'.format(args.k))
            write_sent(sd, output)

    if args.sample:
        z = np.random.normal(size=(args.n, model.args.dim_z)).astype('f')
        sents = decode(z)
        write_sent(sents, os.path.join(args.checkpoint, args.output))

    if args.tsne:
        sents = load_sent(args.data)
        z, l = encode_annotated(sents)
        tsne = TSNE(n_jobs=10)
        res = tsne.fit_transform(z)
        color_map = np.argmax(l, axis=1)
        plt.figure(figsize=(10, 10))
        for cl in range(args.output_size):
            indices = np.where(color_map == cl)[0]
            plt.scatter(res[indices, 0], res[indices, 1], label=cl)
        plt.legend()
        plt.savefig('./img/tsne.png')

    if args.toy_tsne:
        lines, x, y = create_toy_dataset()
        color_map = np.array(y)
        z = encode(x)
        tsne = TSNE(n_jobs=10)
        res = tsne.fit_transform(z)
        print(res.shape)
        plt.figure(figsize=(15, 15))

        for cl in range(8):
            indices = np.where(color_map == cl)
            plt.scatter(res[indices, 0], res[indices, 1], label=cl)
        plt.legend()
        plt.savefig('./img/{}.png'.format(args.output))

    if args.reconstruct:
        sents = load_sent(args.data)
        z = encode(sents)
        sents_rec = decode(z)
        write_z(z, os.path.join(args.checkpoint, args.output+'.z'))
        write_sent(sents_rec, os.path.join(
            args.checkpoint, args.output+'.rec'))

    if args.arithmetic:
        fa, fb, fc = args.data.split(',')
        sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
        za, zb, zc = encode(sa), encode(sb), encode(sc)
        zd = zc + args.k * (zb.mean(axis=0) - za.mean(axis=0))
        sd = decode(zd)
        write_sent(sd, os.path.join(args.checkpoint, args.output))

    if args.interpolate:
        f1, f2 = args.data.split(',')
        s1, s2 = load_sent(f1), load_sent(f2)
        z1, z2 = encode(s1), encode(s2)
        zi = [interpolate(z1_, z2_, args.n) for z1_, z2_ in zip(z1, z2)]
        zi = np.concatenate(zi, axis=0)
        si = decode(zi)
        si = list(zip(*[iter(si)]*(args.n)))
        write_doc(si, os.path.join(args.checkpoint, args.output))

    if args.latent_nn:
        sents = load_sent(args.data)
        z = encode_annotated(sents)
        with open(os.path.join(args.checkpoint, args.output), 'w') as f:
            nn = NearestNeighbors(n_neighbors=args.n).fit(z)
            dis, idx = nn.kneighbors(z[:args.m])
            for i in range(len(idx)):
                f.write(' '.join(sents[i]) + '\n')
                for j, d in zip(idx[i], dis[i]):
                    f.write(' '.join(sents[j]) + '\t%.2f\n' % d)
                f.write('\n')
    
    if args.mean_flip_dist:
        sents = load_sent(args.data)
        z, l = encode_annotated(sents)
        lbls = np.argmax(l, axis=-1)
        with open(os.path.join(args.checkpoint, 'mean_flip_dist'), 'w') as f:
            nn = NearestNeighbors(n_neighbors=args.n).fit(z[:args.m])
            dis, idx = nn.kneighbors(z[:args.m])
            a = 0.0
            a_nn = 0
            for i in range(len(idx)):
                done = False
                refs = []
                f.write(' '.join(sents[i]) + '\n')
                c = 0    
                for j, d in zip(idx[i], dis[i]):
                    c += 1
                    if(not done and abs(lbls[j] - lbls[i]) == 1): #polarity has flipped
                        a += d
                        a_nn += c
                        done = True
                        f.write(' '.join(sents[j]) + '\t%.2f' % d + '\t {} \n'.format(lbls[j]))    
                        break
                f.write('\n')
            f.write('Avg distance for label flip: {}\n'.format(a/args.m))
            f.write('Avg neighbour count for label flip: {}\n'.format(a_nn/args.m))

