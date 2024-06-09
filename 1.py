# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:43:46 2024

@author: pk
"""

import numpy as np
print(np.__version__)
a = 10
#%%
for i in range(10):
    print('iterator: ', i)
#%%
l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in l:
    print('list: ', i)
#%%
def my_iter(r):
    i = 0
    if i < r:
        yield i
        i = i + 1
        
for i in my_iter(10):
    print('my iter', i)
#%%
l = [i*10 for i in range (10) if i % 2 == 0]
print (l)


l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
l_out = []
for i in l:
    if i % 2 == 0:
        l_out.append(i*10)
        
print('l_out=', l_out)
#%%
it = range(10)
print(list(it))
#%%
d= {"imie": "Jan", 'nazwisko' : 'Kowalski', 'wiek':27}
d['imie']
d['imie'].__class__
#%%
T = (1, 2, 3, 4)
l = [1, 2, 3, 4]
#%%
import numpy as np

a=np.array([1, 2, 3,4.5,5], dtype= np.int64)

for i in a:
    print(i)

a + 1
np.sin(a)
#%%
M = np.arange(20).reshape(5,4)
M
M.shape
M[0]
M[0, 1]
M[1:3]
M[:3]
M[1:]
M[1:,2:]
M[1:-1:2, 2:]
#%%
a=np.arrange(100, 2_000_000)
a2 = a[a%2 == 0].copy()
%timeit a.mean()
%timeit a2.mean()