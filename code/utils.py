# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
STUDENT = {'name1': 'Omri Ben Hemo',
           'ID1': '313255242',
           'name2': 'Ben Ganon',
           'ID2': '318731007'
           }

def read_data(fname):
    data = []
    with open(fname, encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().lower().split("\t",1)
            data.append((label, text))
    return data

def text_to_unigrams(text):
    return [c for c in text]

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

TRAIN_UNI = [(l,text_to_unigrams(t)) for l,t in read_data("../data/train")]
DEV_UNI = [(l,text_to_unigrams(t)) for l,t in read_data("../data/dev")]
TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data("../data/train")]
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("../data/dev")]

from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

unic = Counter()
for l,feats in TRAIN_UNI:
    unic.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x,c in fc.most_common(600)])
vocab_uni = set([x for x,c in unic.most_common(600)])
# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}
F_UNI_I = {f:i for i,f in enumerate(list(sorted(vocab_uni)))}
text = "Swear they have a day for anything . National Peanut Butter & Jelly Day ? #DoingTooMuch"
print(L2I)