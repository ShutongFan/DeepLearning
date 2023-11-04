import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from collections import Counter
from itertools import chain
from collections import namedtuple
import numpy as np
import math
from nltk.translate import bleu_score
# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32 # batch size
clip_grad = 5.0 # gradient clipping
valid_niter = 1000 # perform validation after how many iterations
log_every = 200 # show verbouse log every log_every training iterations
vocab_size = 50000
model_save_path = 'model/model.bin'
embed_size=256
hidden_size=256
dropout_rate=0.2
label_smoothing=0.1
uniform_init=0.1 # uniformly initialize all parameters
learning_rate=0.001
patience_top=5 # wait for patience_top iterations to decay learning rate
max_num_trial=5 # terminate training after max_num_trial trials
lr_decay=0.5 # learning rate decay
n_epochs=3
beam_size=5
max_decoding_time_step=70 # maximum number of decoding time steps to unroll the decoding RNN

# Support function for parsing train and test sets
def read_corpus(file_path, source):
    data = []
    for line in open(file_path, encoding='utf-8'):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

