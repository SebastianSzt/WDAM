# -*- coding: utf-8 -*-
#http://fizyka.umk.pl/~mich/kmeans.png

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

np.random.seed(8)
centrs = np.random.rand(3,2)

std = 0.1
num = 80

X = np.vstack([
         np.random.normal(centrs[0], [std, std], (num, 2)),
         np.random.normal(centrs[1], [std, std], (num, 2)),
         np.random.normal(centrs[2], [std, std], (num, 2))
    ])

plt.scatter(X.T[0], X.T[1])
plt.scatter(centrs.T[0], centrs.T[1], marker='x', s=100)
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.show()

#Algorytm k-means
### input: X i k=3
#1 Wylosuj centroidy
c = np.random.rand(3, 2)

#2. oblicz odleglosci miedzy c i X
for iter_no in range(100):
    #D = np.zeros((len(X), len(c)))
    #for i, (x, y) in enumerate(X):
    #    for j in range(len(c)):
    #        cx, cy = c[j]
    #        D[i,j] = np.sqrt((x-cx)**2 + (y-cy)**2)
            
    D = distance.cdist(X,c)
    b = np.argmin(D, axis=1)
    
    c_new = [X[b == i].mean(axis=0) for i in range(len(c))]
    c_new = np.array(c_new)
    
    inertia = np.sqrt(((c_new - c)**2).sum())
    print("iterno=%d, inertia=%.3f" %(iter_no, inertia))
    if inertia < 0.0001: break

    c = c_new
    
    plt.scatter(X.T[0], X.T[1], c=b)
    plt.scatter(c.T[0], c.T[1], marker='x', s=100)
    plt.show()