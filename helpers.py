# -*- coding: utf-8 -*-
import os
from six.moves import xrange
from sys import exit
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy
import math
import gensim.downloader as api


def normalizer(myvector):
    mysum = 0.
    for myvalue in myvector:
        mysum += myvalue * myvalue
    if mysum <= 0.:
        return myvector
    mysum = math.sqrt(mysum)
    newvector = []
    for myvalue in myvector:
        newvector.append(myvalue/mysum)
    return newvector


def emblookup(words, word2vec):
    ret = []
    for w in words:
        w = w.lower()
        if w not in word2vec:
            continue
        ret.append(word2vec[w])
    return ret


def emblookup_verbose(words, word2vec):
    ret = []
    ret_w = []
    for w in words:
        w = w.lower()
        if w not in word2vec:
            continue
        ret.append(word2vec[w])
        ret_w.append(w)
    return ret_w


def line_process(l):
    l = l.strip().split()
    try:
        word = l[0].decode("utf-8").lower()
    except:
        print (l[0])
        return (None, None)
    vals = normalizer([float(v) for v in l[1:]])
    return (word, vals)


def word2vec(emb_path):
    word2vec = {}
    pool = Pool(4)
    with open(emb_path, "r") as f:
        pairs = pool.map(line_process, f.readlines()[1:])
    pool.close()
    pool.join()
    _pairs = []
    for p in pairs:
        if p[0] is not None:
            _pairs.append(p)
    return dict(_pairs)

def load_embeddings(emb):
    model = api.load('glove-wiki-gigaword-50')
    model.init_sims(replace=True)
    emb_dim = model.vectors_norm.shape[1]
    emb_dict = dict(zip(model.index2word, model.vectors_norm))
    emb_vocab = model.index2word
    emb_vectors = model.vectors_norm
    return emb_dict, emb_dim, emb_vocab, emb_vectors


def read_lexicon_file(path, category):
    words = list()
    with open(os.path.join(path, '{}.txt'.format(category)), 'r') as f:
        lines = f.readlines()
    for line in lines:
        words.append(line.strip())
    return words

def load_lexicon(path):
    pos = read_lexicon_file(path, 'pos')
    neg = read_lexicon_file(path, 'neg')
    return pos, neg
