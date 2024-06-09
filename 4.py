# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('iris-data.csv')

df
df.columns

df[df.isnull().values.any(axis=1)]

#Co zrobić w przypadku braku w danych?
df.dropna()

null_idxs = df.isnull().values.any(axis=1)
impute_val = df.dropna()[df['class'] == 'Iris-setosa']['petal_width_cm'].mean()
df.loc[null_idxs, 'petal_width_cm'] = impute_val


sns.pairplot(df, hue='class')


df[df['sepal_length_cm'] < 2.5]

#poprawiamy bledne jednostki
df.loc[df['sepal_length_cm'] < 2.5, 'sepal_length_cm'] *= 100

df['class'].unique
df.loc[df['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
df.loc[df['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'

#%%
from sklearn import datasets, tree, metrics

iris = datasets.load_iris()
iris.data[:10]
iris.target
iris.target_names
