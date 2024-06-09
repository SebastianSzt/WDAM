# -*- coding: utf-8 -*-
#http://fizyka.umk.pl/~mich/text.py

import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
stemer = PorterStemmer()

docs = ["Human machine interface for Lab ABC computer applications",
"A survey of user opinion of computer system response time",
"The EPS user interface management system",
"System and human system engineering testing of EPS",
"Relation of user-perceived response time to error measurement",

"The generation of random, binary, unordered trees",
"The intersection graph of paths in trees",
"Graph minors IV: Widths of trees and well-quasi-ordering",
"Graph minors: A survey"]

docs = [d.lower() for d in docs]
docs = [nltk.word_tokenize(d) for d in docs]

fdist = nltk.FreqDist()

for doc in docs:
    for word in doc:
        fdist[word] +=1
        
fdist.plot()

from nltk.corpus import stopwords
import numpy as np

stop_set = set(stopwords.words('english')).union(set([":", ";", ","]))
stop_set = frozenset(stop_set)


fdist.keys()
words = set(fdist.keys()).difference(stop_set)

word2id = dict(zip(words, range(len(words))))

M = np.zeros(shape=(len(docs), len(words)))
for doc_num, doc in enumerate(docs):
    for word in doc:
        if word in stop_set:
            continue
        M[doc_num, word2id[word]] += 1
        
plt.imshow(M)

from scipy.spatial import distance

distance.cosine(M[0], M[1])
distance.cosine(M[0], M[2])

distance.cosine(M[0], M[5])
distance.cosine(M[0], M[6])

D = distance.pdist(M, metric="cosine")
D = distance.squareform(D)
plt.imshow(D)


u, e, v = np.linalg.svd(M.T, full_matrices = False)
#np.round(u.dot(np.diag(e)).dot(v))
e_trunc = np.zeros(9)
e_trunc[:2] = e[:2]

TD = np.diag(e_trunc).dot(v)[:2]
plt.imshow(np.abs(TD), cmap='gray')

TT = u.dot(np.diag(e_trunc))[:,:2].T
TT.shape

id2word = {v:k for (k,v) in word2id.items()}

t1 = np.abs(TT[0])
t2 = np.abs(TT[1])

[id2word[i] for i in np.argsort(t1)[::-1][:5]]
[id2word[i] for i in np.argsort(t2)[::-1][:5]]

q_term = 'computer'
q_vec = np.zeros(len(word))
q_vec[word2id[q_term]] = 1

q_latent = q_vec.dot(u.dot(np.diag(e_trunc)))[:2]
q_latent
docs_latent = np.diag(e_trunc).dot(v)[:2]
docs_latent.shape
dists = [distance.cosine(q_latent, d) for d in docs_latent]
dists