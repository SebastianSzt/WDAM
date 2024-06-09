# -*- coding: utf-8 -*-
#Projekt do 13.06 
#06.06 nie ma zajęć - odrabiamy zajęcia 10.06

from skimage import color, io, transform, exposure
import numpy as np
import matplotlib.pyplot as plt

I = io.imread("coins/test1.jpg")

I = I.astype(float) / 255

plt.figure();
plt.imshow(I)

#%matplotlib qt

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)

Ihsl = color.rgb2hsv(I)

ax1.imshow(Ihsl[:,:,0])
ax2.imshow(Ihsl[:,:,1])
ax3.imshow(Ihsl[:,:,2])

#%%
from skimage.morphology import binary_dilation, binary_erosion, binary_closing
from skimage.measure import label, regionprops

Ig = (Ihsl[:,:,1] * -1) + 1

plt.imshow(Ig, cmap='gray')

plt.hist(Ig.ravel(), bins=100)
It = Ig > 0.6
plt.imshow(It)
plt.figure(); plt.imshow(binary_closing(It))

It = binary_closing(It)

#%%

L = label(It)
plt.imshow(L)
Icopy = np.zeros(shape = I.shape, dtype=float)
#Icopy[L == 2] = I[L == 2]
regions = regionprops(L)

#reg = regions[0]
#reg.bbox

#plt.imshow(Icopy)

coins_imgs = []
for region in regions:
    if region.area > 100:
        minr, minc, maxr, maxc = region.bbox
        C = I[minr:maxr, minc:maxc]
        coins_imgs.append(C)
        plt.figure()
        plt.imshow(C)

#%%
from skimage.transform import warp_polar

#list(map(np.shape, coins_imgs))

for C in coins_imgs:
    Cp = warp_polar(C, channel_axis=2)
    plt.figure()
    
    plt.imshow(Cp)
    
#%%

C = coins_imgs[1]
C = color.rgb2hsv(C)
Cp = warp_polar(C, channel_axis=2)
Cp.shape
Cp.mean(axis=0).shape
plt.figure(); plt.imshow(Cp)
plt.figure();
plt.plot(Cp.mean(axis=0).T[0], label = 'hue')
plt.plot(Cp.mean(axis=0).T[1], label = 'saturation')
plt.plot(Cp.mean(axis=0).T[2], label = 'value')

_,w, _ = Cp.shape
plt.axvline(0.4 * w, c = 'r')
plt.axvline(0.7 * w, c = 'r')
plt.legend()

#%%
#Dla każdego obrazka C zamienić go na wektor 2 liczb hue w srodku i hue na zewnątrz
def segment_img():
    I = io.imread("coins/test1.jpg")
    I = I.astype(float) / 255
    Ihsl = color.rgb2hsv(I)
    It = Ig > 0.5
    It = binary_closing(It)
    L = label(It)
    plt.figure()
    plt.imshow(L)
    return L

def coinimgs2features(coin_imgs):
    f = []
    for C in coins_imgs:
        C = color.rgb2hsv(C)
        Cp = warp_polar(C, channel_axis=2)
        _,w, _ = Cp.shape
        hue_channel = Cp.mean(axis=0).T[0]
        h1 = hue_channel[:int(0.4*w)]
        h2 = hue_channel[int(0.4*w):int(0.7*w)].mean()
        f.append([h1, h2])
    return np.array(f)
    
#%%
F = []

I = io.imread("coins/5b.jpg")
I = I.astype(float) / 255
L = segment_img(I)
img_coins = segm2coinsimg(L, I)
f = coinimgs2features(img_coins)

F.append(f)

F = np.vstack(F )

#%%
plt.scatter(F.T[0][:], F.T[])