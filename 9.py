# -*- coding: utf-8 -*-
from skimage import data, exposure, color
import matplotlib.pyplot as plt
import numpy as np

img = data.camera()

#plt.imshow(img, cmap='gray')
#plt.imshow(img[110:160, 230:280], cmap='gray')

img = img.astype(np.float64) / 255
plt.imshow(img, cmap='gray')

plt.colorbar()

img.shape


img = data.coffee() 
img.shape
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols = 3)
ax1.imshow(img[:,:,0])
ax2.imshow(img[:,:,1])
ax3.imshow(img[:,:,2])

plt.imshow(img)
#plt.imshow(img.mean(axis=2), cmap='gray')

#%%
img = data.coffee()

img = color.rgb2hsv(img)

img[:,:,2] *= 0.5

img = color.hsv2rgb(img)

plt.imshow(img)

#%%

from skimage import color, io, transform, exposure
import numpy as np
import matplotlib.pyplot as plt

I = io.imread("coins/test1.jpg")

I = transform.resize(I, (800, 600, 3))
I = color.rgb2gray(I)

plt.figure()
plt.imshow(I, cmap='gray')


plt.figure(); plt.hist(I.flatten(), bins = 100)

I = exposure.equalize_adapthist(I)
plt.figure();
plt.imshow(I, cmap='gray')


