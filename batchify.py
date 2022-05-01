import torch
from torch._C import dtype
import torch.nn.functional as F
import numpy as np

def get_batch(x, vocab, device):
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

def get_batches(data, vocab, batch_size, device, no_shuffle=False, constant_batch=False):
    if(not no_shuffle):
        # shuffle in increasing order of sentence length
        order = range(len(data))
        z = sorted(zip(order, data), key=lambda i: len(i[1]))
        order, data = zip(*z)
    else:
        order = range(len(data))

    batches = []
    i = 0

    if(constant_batch): #all batches will have leading lenght = batch_size
        while(i + batch_size < len(data)):
            batches.append(get_batch(data[i: i+batch_size], vocab, device))    
            i += batch_size
    else: #all instances in batch of same size
        while i < len(data):
            j = i
            while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
                j += 1
            batches.append(get_batch(data[i: j], vocab, device))
            i = j
    return batches, order


def get_batch_annotated(x, vocab, device, cbow=None, num_classes=2):
    go_x = []
    targets = []
    cbow_tar = torch.zeros(len(x), (len(cbow) if(cbow != None) else 1)) #BC
    max_len = max([len(s) for s in x])
    for i, s in enumerate(x):
        s_idx = [vocab.word2idx[s[i]] if s[i] in vocab.word2idx else vocab.unk for i in range(len(s) - 2)]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        targets.append((int)(s[-1]))    
        if(cbow is not None):
            for word in s:
                if word in cbow:
                    cbow_tar[i][cbow[word]] += 1
        
    cbow_max, _ = torch.max(cbow_tar, dim=-1) #B
    cbow_tar = torch.tensor(cbow_tar)/cbow_max.unsqueeze(dim=-1)
    targets = torch.tensor(F.one_hot(torch.tensor([targets]), num_classes=num_classes), dtype=torch.float).squeeze()
    return torch.LongTensor(go_x).t().contiguous().to(device), targets.contiguous().to(device), cbow_tar.contiguous().to(device)

def get_batches_annotated(data, vocab, batch_size, device, cbow=None, num_classes=2, constant_batch=False):
    #sort data in increasing order of sentence length
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)   
    batches = []
    i = 0

    if(constant_batch): #all batches will have leading lenght = batch_size  
        while(i + batch_size < len(data)):
            batches.append(get_batch_annotated(data[i: i+batch_size], vocab, device))    
            i += batch_size
    else:
        while i < len(data):
            j = i
            while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
                j += 1
            
            batches.append(get_batch_annotated(data[i: j], vocab, device, cbow=cbow, num_classes=num_classes))
            i = j

    return batches , order

