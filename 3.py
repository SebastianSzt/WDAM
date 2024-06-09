import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

index = pd.date_range("1-1-2020", periods=8, freq='d')
s = pd.Series(np.random.randn(5), index=['a','b','c','d','e'])
s.index

s.values

s_cat = pd.Categorical(['test', 'train', "test", 'train', "test", 'train', "test", 'train'])

df = pd.DataFrame(
    data = np.random.randn(8,3),
    index = index,
    columns = ["A","B","C"]
    )

df.head()
df.tail()
df.describe()

d = np.random.randn(8)
df = df.assign(D = d)

df.shape
s_cat.shape
df = df.assign(cat = s_cat)

df[df > 0] = 1.0
df["A"]
df.loc['2020-01-03':'2020-01-05']
df.iloc[1:4]
df.loc['2020-01-03':'2020-01-05', ['A', 'B']]

s_mean = df.mean()
s_mean.__class__

df2 = df.copy()

pd.concat([df, df2])

df
df.groupby(by='cat').mean()

df_train = df[df['cat'] == 'train']
df_test = df[df['cat'] == 'test']

df_train
df_test

#%%
df = pd.DataFrame(
    data = np.random.randn(800,3),
    columns = ["A","B","C"]
    )
df = df.cumsum()

df.plot()
