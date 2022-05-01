import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_anneal_weight, get_glove

from noise import noisy, embd_noise

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args    
        if(args.glove):
            weights = torch.FloatTensor(get_glove(vocab))            
            self.embed = nn.Embedding.from_pretrained(weights)
        else:
            self.embed = nn.Embedding(vocab.size, args.dim_emb)
        self.proj = nn.Linear(args.dim_h, vocab.size)
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

class DAE(TextModel):
    """Denoising Auto-Encoder/ Parent Class of VAE/AAE/VanillaAE"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.drop = nn.Dropout(args.dropout) 
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)
        self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
        self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb)
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
        self.step_count = 0
        self.ann_weight = 0.0
        self._zeta = 0.0

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()
    

    def encode(self, input, args, is_train=False):
        embds = self.embed(input) #LBE
        if(args.embd_noise and is_train):
            self.ann_weight = get_anneal_weight(self.step_count/args.log_interval, args) if args.fn is not None else 1.0
            self._zeta = args.zeta * self.ann_weight
            noise = embd_noise(embds, noise_type=args.noise_type, zeta=self._zeta)
            embds  = embds + noise
            input = self.drop(embds)
        else: 
            input = self.drop(self.embed(input))        

        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h), h

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input, args, is_train=False):
        _input = noisy(self.vocab, input, *self.args.noise) if(is_train) else input
        mu, logvar, _ = self.encode(input=_input, args=args, is_train=is_train)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec']

    def get_anneal_info(self):
        return {"steps": self.step_count, 
                "weight": self.ann_weight, 
                "zeta": self._zeta}

    def autoenc(self, inputs, targets, args, is_train=False):
        _, _, _, logits = self(inputs, args, is_train)
        self.step_count += 1
        return {'rec': self.loss_rec(logits, targets).mean()}
    
    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

    def nll_is(self, inputs, targets, args, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar, _ = self.encode(inputs, args)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is


class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, args, targets, is_train=False):
        mu, logvar, _, logits = self(input=inputs, args=args, is_train=is_train)
        self.step_count += 1
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}


class AAE(DAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
            nn.Linear(args.dim_d, 1), nn.Sigmoid())
        self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy(self.D(zn), ones)
        loss_g = F.binary_cross_entropy(self.D(z), ones)
        return loss_d, loss_g

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_adv * losses['adv'] + \
            self.args.lambda_p * losses['|lvar|']

    def autoenc(self, inputs, targets, args, is_train=False):
        _, logvar, z, logits = self(inputs, args, is_train)
        loss_d, adv = self.loss_adv(z)
        self.step_count += 1
        return {'rec': self.loss_rec(logits, targets).mean(),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}
    

    def step(self, losses):
        super().step(losses)

        self.optD.zero_grad()
        losses['loss_d'].backward()
        self.optD.step()

class VanillaAE(DAE):
    """Deterministic Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.h2ls = nn.Linear(args.dim_h*2, args.dim_z)

    def encode(self, input, args, is_train=False):
        embds = self.embed(input) #LBE
        if(args.embd_noise and is_train):
            self.ann_weight = get_anneal_weight(self.step_count/args.log_interval, args) if args.fn is not None else 1.0
            self._zeta = args.zeta * self.ann_weight
            noise = embd_noise(embds, noise_type=args.noise_type, zeta=self._zeta)
            embds  = embds + noise
            input = self.drop(embds)
        else: 
            input = self.drop(self.embed(input))        

        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2ls(h), None, h

    def forward(self, input, args, is_train=False):
        _input = noisy(self.vocab, input, *self.args.noise) if(is_train) else input
        z, _, _ = self.encode(_input, args, is_train=is_train)
        logits, _ = self.decode(z, input)
        return z, logits

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, args, is_train=False):
        z, logits = self(inputs, args, is_train)
        self.step_count += 1
        return {'rec': self.loss_rec(logits, targets).mean()}

class Classifier(TextModel):
    """Classifier for Style transfer accuracy metric"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

        self.drop = nn.Dropout(args.dropout) 
        self.output_size = args.output_size 
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers, bidirectional=True)            
        self.h2out = nn.Linear(args.dim_h*2, self.output_size)
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    def flatten(self):
        self.E.flatten_parameters()
        

    def forward(self, input):
        input = self.drop(self.embed(input)) #BLE
        _, (h, _) = self.E(input) #2*BH
        h = torch.cat([h[-2], h[-1]], 1) #B(H*2)    
        out = self.h2out(h) #BO
        out = nn.Softmax(dim=1)(out)
        return out

    def compute_acc(self, yhat, targets):
        yhat = yhat.squeeze()
        targets = targets.squeeze()
        n_samples = targets.size(0)
        yhat = torch.argmax(yhat, dim=-1)
        targets = torch.argmax(targets, dim=-1)    
        return yhat[yhat == targets].size(0)/n_samples


    def loss_ce(self, yhat, targets):
        criterion = nn.BCELoss(reduction='none')
        targets = targets.squeeze()
        yhat = yhat.squeeze()
        loss = criterion(yhat, targets)
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['ce']

    def autoenc(self, inputs, targets):
        outs = self(inputs)
        return {'ce': self.loss_ce(outs, targets).mean()}, outs

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        self.opt.step()
