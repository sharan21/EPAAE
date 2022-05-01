import numpy as np
import torch

def word_shuffle(vocab, x, k):   # slight shuffle such that |sigma[i]-i| <= k
    base = torch.arange(x.size(0), dtype=torch.float).repeat(x.size(1), 1).t()
    inc = (k+1) * torch.rand(x.size())
    inc[x == vocab.go] = 0     # do not shuffle the start sentence symbol
    inc[x == vocab.pad] = k+1  # do not shuffle end paddings
    _, sigma = (base + inc).sort(dim=0)
    return x[sigma, torch.arange(x.size(1))]

def word_drop(vocab, x, p):     # drop words with probability p
    x_ = []
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True  # do not drop the start sentence symbol
        sent = [w for j, w in enumerate(words) if keep[j]]
        sent += [vocab.pad] * (len(words)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def word_blank(vocab, x, p):     # blank words with probability p
    blank = (torch.rand(x.size(), device=x.device) < p) & \
        (x != vocab.go) & (x != vocab.pad)
    x_ = x.clone()
    x_[blank] = vocab.blank
    return x_

def word_substitute(vocab, x, p):     # substitute words with probability p
    keep = (torch.rand(x.size(), device=x.device) > p) | \
        (x == vocab.go) | (x == vocab.pad)
    x_ = x.clone()
    x_.random_(vocab.nspecial, vocab.size)
    x_[keep] = x[keep]
    return x_

def noisy(vocab, x, drop_prob, blank_prob, sub_prob, shuffle_dist):
    if shuffle_dist > 0:
        x = word_shuffle(vocab, x, shuffle_dist)
    if drop_prob > 0:
        x = word_drop(vocab, x, drop_prob)
    if blank_prob > 0:
        x = word_blank(vocab, x, blank_prob)
    if sub_prob > 0:
        x = word_substitute(vocab, x, sub_prob)
    return x

def embd_noise(embds, noise_type='hollow', zeta='0.0'):
    embeds_magn = torch.norm(embds, dim=-1) #LB
    noise = torch.rand_like(embds) #LBE
    if(noise_type == 'hollow'):
        noise = (zeta * embeds_magn).unsqueeze(-1) * torch.nn.functional.normalize(noise, p=2.0, eps=1e-12, dim=-1) #LBE
    elif(noise_type == 'centered-gau'): #gaussian hypersphere, mu = 0, var = (zeta/3)^2
        noise_magn = zeta/3 * torch.randn(embeds_magn.size(0), embeds_magn.size(1), device=embds.device)#LB
        noise = (noise_magn * embeds_magn).unsqueeze(-1) * torch.nn.functional.normalize(noise, p=2.0, eps=1e-12, dim=-1) #LBE
    elif(noise_type == 'uniform'): #uniform hypersphere in [0, zeta]
        noise_magn = zeta * torch.rand(embeds_magn.size(0), embeds_magn.size(1), device=embds.device) #LB
        noise = (noise_magn * embeds_magn).unsqueeze(-1) * torch.nn.functional.normalize(noise, p=2.0, eps=1e-12, dim=-1) #LBE
    elif(noise_type == 'shifted-gau'): #gaussian hypersphere, mu = zeta, var = 1                
        noise_magn = zeta + torch.randn(embeds_magn.size(0), embeds_magn.size(1), device=embds.device) #LB
        noise = (noise_magn * embeds_magn).unsqueeze(-1) * torch.nn.functional.normalize(noise, p=2.0, eps=1e-12, dim=-1) #LBE                
    else:
        exit("args.noise_type is not a valid option")
    return noise
