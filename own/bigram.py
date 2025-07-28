import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda_is_available() else 'cpu'
eval_iters = 200
train_split = 0.9

# load the dataset as a string
with open('../dickens/combined.txt', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create mappings chars <-> ints
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])

# split into training and testing split
data = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(0.9*len(data))
data_train = data[:split_idx]
data_val = data[split_idx:]

# load random batches
def get_batch(split):
    data = data_train if split == 'train' else data_val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])