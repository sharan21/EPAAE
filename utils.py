
import random
import os
import math
import numpy as np
import torch
import pickle

class ClassifierEvaluationDataset(torch.utils.data.Dataset):
	def __init__(self, encodings):
		self.encodings = encodings

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		return item

	def __len__(self):
		return len(self.encodings.input_ids)

class AverageMeter(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.cnt = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.cnt += n
        self.sum += val * n
        self.avg = self.sum / self.cnt

def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs, targets in batches:
            losses = model.autoenc(inputs, targets, args)
            for k, v in losses.items():
                meters[k].update(v.item(), inputs.size(1))
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters
	
def shuffle_in_unison(a, b):
	c = list(zip(a, b))
	random.shuffle(c)
	a, b = zip(*c)
	return a, b

def get_anneal_weight(step, args):
		assert(args.fn != 'none')
		if args.fn == 'logistic':
			return 1 - float(1/(1+np.exp(-args.k*(step-args.tot_steps))))    
		elif args.fn == 'sigmoid':
			return 1 - (math.tanh((step - args.tot_steps * 1.5)/(args.tot_steps / 3))+ 1)
		elif args.fn == 'linear': 
			return min(1, step/args.tot_steps)
		else:
			exit("wrong option in args.fn")

def clean_line(line):
	line = line.lower().replace(',','').replace('.','').replace('\'', '').replace('!','').replace(')','').replace('(','').replace(';','').replace('-', ' ').replace('/', ' ').replace('%','').replace('?','').replace('"', '')
	return line

def set_seed(seed):     # set the random seed for reproducibility
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

def get_cbow(vocab, paths=["data/lexicon/neg_words.txt", "data/lexicon/pos_words.txt"]):
	count = 0
	cbow = dict(vocab.word2idx)
	for path in paths:
		with open(path) as f:
			for line in f:
				word = line[:-1]
				count += 1
				if word in cbow:
					del cbow[word]
		# print("Found {} sentiment words in {}".format(count, path))
	# print("Original w2i size: {}".format(len(vocab.word2idx)))
	return { w: it  for it, (w, i) in enumerate(cbow.items())}

def create_glove(embd_dim=300):
	path = 'glove/glove.6B.{}d.txt'.format(embd_dim)
	print("Creating glove embds from {}".format(path))
	glove = {}    
	with open(path, 'rb') as f:
		for idx, l in enumerate(f):
			if(idx % 100000 == 0):
				print(idx)
			line = l.decode().split()
			glove[line[0]] = np.array(line[1:]).astype(np.float)
	pickle.dump(glove, open('glove/glove-{}.pkl'.format(embd_dim), 'wb'))

def load_glove(embd_dim=300):
	path = 'glove/glove-{}.pkl'.format(embd_dim)
	if(os.path.exists(path)):
		glove = pickle.load(open(path, 'rb'))
	else:
		exit("did not find glove in given path")
	return glove

def get_glove(vocab, embd_dim=300):
	glove = load_glove(embd_dim)
	glove_mat = np.zeros((len(vocab.word2idx), glove[next(iter(glove))].size))
	for word, idx in vocab.word2idx.items():
		glove_mat[idx] = glove[word] if word in glove else np.random.normal(scale=0.6, size=(glove[next(iter(glove))].size, ))
	return glove_mat
	

def strip_eos(sents):
	return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
		for sent in sents]

def load_sent(path, split=True):
	sents = []
	with open(path) as f:
		for line in f:
			if(split):
				sents.append(line.split())
			else:
				sents.append(line)
	return sents

def write_sent(sents, path):
	with open(path, 'w') as f:
		for s in sents:
			l = ' '.join(s) + '\n'
			if l == '\n':
				f.write('<blank> \n')	
			else:
				f.write(l)

def write_doc(docs, path):
	with open(path, 'w') as f:
		for d in docs:
			for s in d:
				f.write(' '.join(s) + '\n')
			f.write('\n')

def write_z(z, path):
	with open(path, 'w') as f:
		for zi in z:
			for zij in zi:
				f.write('%f ' % zij)
			f.write('\n')

def logging(s, path, print_=True):
	if print_:
		print(s)
	if path:
		with open(path, 'a+') as f:
			f.write(s + '\n')
	
def lerp(t, p, q):
	return (1-t) * p + t * q

# spherical interpolation https://github.com/soumith/dcgan.torch/issues/14#issuecomment-199171316
def slerp(t, p, q):
	o = np.arccos(np.dot(p/np.linalg.norm(p), q/np.linalg.norm(q)))
	so = np.sin(o)
	return np.sin((1-t)*o) / so * p + np.sin(t*o) / so * q

def interpolate(z1, z2, n):
	z = []
	for i in range(n):
		zi = lerp(1.0*i/(n-1), z1, z2)
		z.append(np.expand_dims(zi, axis=0))
	return np.concatenate(z, axis=0)

def shuffle_two_lists(list1, list2):
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    return res1, res2

def flip(p=0.008):
    return True if random.random() < p else False


if __name__ == "__main__":
	pass

