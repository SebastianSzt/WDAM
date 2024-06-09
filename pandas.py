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

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pobierz ze strony https://www.kaggle.com/PromptCloudHQ/imdb-data/version/1 plik IMDB-Movie-data.csv

#tutorial:
#df2 = pd.DataFrame(
#    {
#        "A": 1.0,
#        "B": pd.Timestamp("20130102"),
#        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
#        "D": np.array([3] * 4, dtype="int32"),
#        "E": pd.Categorical(["test", "train", "test", "train"]),
#        "F": "foo",
#    }
#)

#1. Wczytaj plik jako dataframe
df = pd.read_csv('imdb.csv', index_col=0)
df.columns
df.dtypes
df
#2. Narysuj histogram ocen
df.hist('Metascore')
#3. posortuj wedlug kolumn Rating i wypisz pierwszych 10
df.sort_values(by='Rating').head(10)

#4. Pokaz 10% najwyzej ocenionych filmow wedlug Metascore i Rating u IMDB
df.sort_values(by=['Metascore', 'Rating'], ascending=False).head(int(len(df.index) * 0.1))
#5. Wypisz filmy Ridley Scotta
df[df['Director'] == 'Ridley Scott']['Title']

#6. Wylicz ile filmow rocznie mialo premiere i zrob wykres
df['Year'].value_counts().plot()

#7. Posortuj Rezyserow wzgledem sredniej oceny
df.groupby('Director')['Rating'].mean().sort_values(ascending=False)

df.groupby('Director').mean('Rating').reset_index().sort_values('Rating', ascending=False)['Director']