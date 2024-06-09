# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:05:44 2024

@author: pk
"""

#http://fizyka.umk.pl/~mich/imgw.py
#http://fizyka.umk.pl/~mich/meteo.csv.gz

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(0)

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

n_samples = 30
X = np.sort(np.random.rand(n_samples))
err = np.random.randn(n_samples) * 0.1
y = true_fun(X) + err

X_range  = np.linspace(0, 1, 50)

plt.plot(X_range, true_fun(X_range), label = 'true')
plt.plot(X,y,'go', label='observation')

poly_deg = 2

p_coefs = np.polyfit(X, y, poly_deg)
f = np.poly1d(p_coefs)

y_reg = f(X_range)
plt.plot(X_range, y_reg, label = "reg")

plt.legend(y, f(X))

plt.title("r2=%.2f" %r2_score(y, f(X)))

#%%
import pandas as pd

df = pd.read_csv('meteo.csv', parse_dates=['date'])

df['Nazwa stacji'].unique()

df_torun = df[df['Nazwa stacji'] == 'TORUŃ']
df_torun = df_torun.set_index(df_torun.date)

#plt.figure(figsize=(10,2))
#df_torun['Maksymalna temperatura dobowa'].plot()

df_torun_y = df_torun.groupby(by=df_torun.index.year).mean()
df_torun_y = df_torun_y[df_torun_y.index < 2024]
df_torun_y['Maksymalna temperatura dobowa'].plot()


#Policz ile stopni rocznie rosnie temperatura w toruniu - wpisać model liniowy i z teg wspolczynnika odczytac ile rocznie bedzie wzrastac temperatura
#sklearn linear regression
#%%
from sklearn.linear_model import LinearRegression

# Przekształcamy rok do postaci numerycznej, ponieważ regresja liniowa wymaga danych numerycznych
X = df_torun_y.index.values.reshape(-1, 1)

# Temperatura maksymalna rocznie
y = df_torun_y['Maksymalna temperatura dobowa'].values

# Inicjujemy model regresji liniowej
model = LinearRegression()

# Dopasowujemy model do danych
model.fit(X, y)

# Współczynnik kierunkowy prostej (czyli współczynnik wzrostu temperatury rocznie)
wzrost_roczny = model.coef_[0]

print("Temperatura rośnie średnio o {:.2f} stopnia na rok.".format(wzrost_roczny))
