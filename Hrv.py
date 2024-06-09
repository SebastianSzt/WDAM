# -*- coding: utf-8 -*-
#%%

import os
import numpy as np
import matplotlib.pyplot as plt

rr_data = []
for fn in sorted(os.listdir('data')):
    rr = open('./data/' + fn).readlines()
    rr = [float(rri.strip()) for rri in rr]
    rr = np.array(rr)
    rr_data.append(rr)

#%%

for i, rr_serie in enumerate(rr_data):
    plt.figure(figsize=(20,2))
    plt.plot(rr_serie, c='b' if i>5 else 'r')
    
    
#ADNN SDRR pNN50 - hrv metrics

def adnn(rr):
    return np.mean(rr)

def sdrr(rr):
    return np.std(rr)

def pNN50(rr):
    d_rr = rr[1:] - rr[:-1]
    return float(np.sum(d_rr > 0.05)) / len(d_rr)

#%%

import pandas as pd

data = []

for s_i in range (10):
    rr = rr_data[s_i]
    row = {
        'age' : 'old' if s_i < 5 else 'young',
        'adnn' : adnn(rr),
        'sdrr' : sdrr(rr),
        'pNN50' : pNN50(rr),
        }
    data.append(row)

df = pd.DataFrame(data)
df

#%%
import seaborn as sns

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,9), ncols=3)
sns.violinplot(data = df, y='adnn', hue = 'age', split=True, ax = ax1)
sns.violinplot(data = df, y='sdrr', hue = 'age', split=True, ax = ax2)
sns.violinplot(data = df, y='pNN50', hue = 'age', split=True, ax = ax3)

#%%

rr = rr_data[5]
plt.figure()
plt.scatter(rr[1:,], rr[:-1], s = 0.5)

rr = rr_data[0]

plt.scatter(rr[1:,], rr[:-1], s = 0.5)

#%%

from scipy import optimize
import sympy

def cost(u, x,y):
    A, B, C, D, E, F = u
    err = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
    return np.mean(err**2)

rr = rr_data[2]
x, y = rr[1:], rr[:-1]
res = optimize.minimize(cost, x0 = [1,1,1,1,1,1], args = (x, y), method='TNC')

A, B, C, D, E, F = res['x']

l = np.linspace(-30, 30)
xx, yy = np.meshgrid(l, l)

I = A*xx**2 + B*xx*yy + C*yy**2 + D*xx + E*yy + F


plt.imshow(I, cmap='jet')