# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:34:15 2019

@author: Personal
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons,Slider
d=cv2.imread('MrBean.jpg',cv2.IMREAD_GRAYSCALE)
def quantise(n):
    if(n==8):
        e=d
    else:
        e=d/2**(8-n)
        e=np.array(e,dtype=int)
        e=np.array(e,dtype=float)
        e=(e+0.5)*2**(8-n)
    return e
e=quantise(1)
plt.figure()
plt.subplot(121)
plt.imshow(d,cmap='gray',vmin=0, vmax=255)
plt.title('MrBean'),plt.xticks([]),plt.yticks([])
plt.subplot(122)
quant=plt.imshow(e, cmap='gray',vmin=0, vmax=255)
plt.title('QUANTISD IMAGE'),plt.xticks([]),plt.yticks([])
selectn=RadioButtons(plt.axes([0.01,0.25,0.10,0.5]),('n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8'),active=0)
plt.figure()
f=np.arange(8)
f=np.array(f,dtype=float)
n=np.arange(8)
for i in range(1,9,1):
    e=quantise(i)
    f[i-1]=20*np.log10((255*np.sqrt(671*540))/np.linalg.norm(d-e))
    n[i-1]=i
error=plt.plot(n,f)
plt.title('PSQNR')#,plt.xticks([]),plt.yticks([])
print(f)
def go(label):
    n=int(label[2],10)
    e=quantise(n)
    quant.set_data(e)
   # Z.set_data(z)
    plt.draw()
selectn.on_clicked(go)
plt.show()