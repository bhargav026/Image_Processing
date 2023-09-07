# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:31:56 2019

@author: Personal
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
b=cv.imread('BaboonGrayscale.jpg', cv.IMREAD_GRAYSCALE)
def avgfilter(n,L):
    n=np.array(n, dtype='float')
    bb=np.zeros((b.shape[0],b.shape[1]), dtype='float')
    for i in range(b.shape[0], 2*b.shape[0],1):
        n[i][b.shape[1]]=np.sum(n[i-L:i+L+1,b.shape[1]-L:b.shape[1]+L+1])/(2*L+1)**2
        for j in range(b.shape[1]+1,2*b.shape[1],1):
            n[i][j]=n[i][j-1]+(np.sum(n[i-L:i+L+1,j+L:j+L+1])-np.sum(n[i-L:i+L+1,j-L:j-L+1]))/(2*L+1)**2
    bb=n[b.shape[0]:2*b.shape[0],b.shape[1]:2*b.shape[1]]
    #bb=bb/((2*L+1)*(2*L+1))
    print(bb)
    return bb
l=5      
a=np.pad(b,b.shape[0],'reflect')
c=avgfilter(a,int(l/2))
plt.figure()
plt.subplot(121)
plt.imshow(a, cmap='gray', vmin=0, vmax=255)
plt.subplot(122)
plt.imshow(c, cmap='gray', vmin=0, vmax=255)
#filterrad=RadioButtons(plt.axes([0.05,0.2,0.15,0.2]),('1.zero','2.mirror','3.periodic'),active=1)
ax1 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
length = Slider(ax1, 'length', 3,19, valinit=5,valstep=2)
#def go(label):
#    
#    global p
#    global a
#    global l
#    p= int(label[0])
#    c=avgfilter(a,int(l/2))
#    plt.subplot(122)
#    plt.imshow(c, cmap='gray', vmin=0, vmax=255)
#    plt.show()

def go1(label):
    global l
    global a
    l=int(label)
    c= avgfilter(a,int(l/2))
    plt.subplot(122)
    plt.imshow(c, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
#filterrad.on_clicked(go)
length.on_changed(go1)