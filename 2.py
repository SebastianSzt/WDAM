#%%
import numpy as np

a = np.arange(9)

print(a)

M = a.reshape(3,3)

M[:, 0]

#%%

import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)

plt.plot(x, y, 'bo-', label="sin(0..2pi)")
plt.grid()
plt.legend()

#%%
import matplotlib.pyplot as plt

 a = np.array(10_000).reshape(100,100)
 plt.imshow(M)
 
 #%%
 #Narysuj za pomoca funkcji imshow wartosci funkcjonału F(x,y) = X^2 + (-y - pierwiastek z trzech x^2)^2 = 1

 def M(x,y):
     return x**2 + (-y - (x**2) ** (1/3)) ** 2 - 1
 
# M = np.zeros(shape=(100,100))
# for i in range(100):
#     for j in range(100):
#         M[i,j] = H(i,j)
        
l = np.linspace(-10, 10, 100)
XX, YY = np.meshgrid(l,l)
##XX,YY.shape
##XX[0][0], YY[0][0]

#H(np.arrange(-10,10), np.arange(-10, 10))
#H(XX,YY).shape

M = H(XX, YY)

plt.imshow(M, cmap='gist_ncar')
plt.colorbar()

#%%
import numpy as np
import matplotlib.pyplot as plt

#http://fizyka.umk.pl/~mich/numpy_basics.py
#Zadanie 1
a = np.random.randn(100).astype(np.int64)
(a == 0).all()
#Zadanie 2
None == None
a is not None
np.isnan(a).all()
#Zadanie 2
a, b = np.random.randn(10_000), np.random.randn(10_000)
np.isclose(a, b, atol = 0.5) #.sum()
#Zadanie 4
a = [0] * 10 + [5] * 10 + [10] * 10
a = np.array(a)

np.random.shuffle(a)
a

###########################

np.concatenate([np.ones(10)*0, np.ones(10)*5, np.ones(10)*10])
np.random.shuffle(a)
a

###########################

a = np.hstack([np.full(10,0), np.full(10,5), np.full(10,10)])
np.random.shuffle(a)
a

#Zadanie 5
np.arrange(30,71)
np.linspace(30, 70,41).astype(int)

#Zadanie 6
np.linspace(0, np.pi, 50)

#Zadanie 7
M = np.zeros((6,6))
M[np.arange(6), np.arange(6)] = 1
M[np.diag_indices_from(M)] = 1
np.eye(6)

#Zadanie 8
I = np.random.normal(size=(100,100))
plt.imshow(I)

#Reszta w domu 5 zadan

#Zadanie 9
#Napisz program, który obliczy iloczyn skalarny dwóch wektorów
import numpy as np
import matplotlib.pyplot as plt
wektor1 = np.array([1, 2, 3])
wektor2 = np.array([4, 5, 6])
print(np.dot(wektor1, wektor2))

#Zadanie 10
#Napisz program, który stworzy macierz 10x10 z losowymi wartościami, i wartością 1 "na brzegach"
import numpy as np
macierz = np.random.rand(10, 10)
macierz[0, :] = 1  # górny brzeg
macierz[:, 0] = 1  # lewy brzeg
macierz[-1, :] = 1  # dolny brzeg
macierz[:, -1] = 1  # prawy brzeg
print(macierz)

#Zadanie 11
#Napisz program, który stworzy macierz 10x10 z losowymi wartosciami typu int, i zastąpi zerami elementy które nie są nieparzyste
import numpy as np
macierz = np.random.randint(0, 100, (10, 10))
macierz[macierz % 2 == 0] = 0
print(macierz)

#Zadanie 12
#Policz sumę 'po-wierszach' i 'po-kolumnach' macierzy z poprzedniego zadania
import numpy as np
#'po-wierszach'
suma_po_wierszach = np.sum(macierz, axis=1)
#'po-kolumnach'
suma_po_kolumnach = np.sum(macierz, axis=0)
print("Suma po wierszach:", suma_po_wierszach)
print("Suma po kolumnach:", suma_po_kolumnach)

#Zadanie 13
#Zapisz i wczytaj powyższą macierz na dysk
import numpy as np
np.save('macierz.npy', macierz)
macierz_wczytana = np.load('macierz.npy')
print("Wczytana macierz:\n", macierz_wczytana)

#Zadanie 14
#Napisz program, który stworzy nową macierz na podstawie tej z poprzedniego zadania tak aby suma elementów wierszy była rosnąca
import numpy as np
suma_po_wierszach = np.sum(macierz, axis=1)
indeksy = np.argsort(suma_po_wierszach)
nowa_macierz = macierz[indeksy]
print("Nowa macierz:\n", nowa_macierz)
