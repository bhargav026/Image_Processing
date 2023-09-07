# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:26:17 2019

@author: JAYATEJA
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider
import cv2 as cv
from PIL import Image
from scipy import signal
from numpy import linalg as LA
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.35)


threshold=0.9
img=cv.imread('Nanoparticles.jpg',0)

template=cv.imread('NanoTemplate.jpg',0)
plt.subplot(121)
plt.imshow(img, cmap='gray', vmin =0, vmax =255 )
plt.show()
plt.title('Input image')
plt.xticks([])
plt.yticks([])

#img=np.array([[1,1,1,1,1,1,1],[1,1,1,1,10,21,13],[1,1,1,1,111,123,251],[1,2,10,10,10,2,1], [2,3,10,10,10,1,3],[152,175,10,10,10,45,91],[121,111,45,15,187,147,1],[115,211,101,151,19,154,121]])
#template=np.array([[1,1,1],[1,1,1],[1,1,1]])
M,N=img.shape
M1,N1=template.shape
template_norm=template/LA.norm(template)
dummy=np.zeros([M-M1+1,N-N1+1])
dummy1=np.zeros([M-M1+1,N-N1+1])
mask=np.zeros([M1,N1])

for i in range(0,M-M1+1):
    for j in range(0,N-N1+1):
        mask=img[i:i+M1,j:j+N1]
        mask_norm=mask/LA.norm(mask)
        dummy[i,j]=np.sum(mask_norm*template_norm)
       
dummy1=255*(dummy>threshold)       
plt.subplot(122)
plt.imshow(dummy1, cmap='gray', vmin =0, vmax =255 )
plt.show()
plt.title('Output image')
plt.xticks([])
plt.yticks([])
print('No.of particles ')
print(np.sum(1*(dummy>threshold)))
#for i range (1,5):
#    for j in range(1,5):
        
#x=np.array([[1,2,3],[1,2,3],[1,2,3]])
#y=np.array([[1,1,1],[1,1,1],[1,1,1]])
#z=np.array([[10,10,10],[10,10,10],[10,10,10]])
#corr = signal.correlate2d(x,y)
#print(corr)
#corr1 = signal.correlate2d(x,z)
#print(corr1)
#
#
#x1=(x)/LA.norm(x)
#y1=(y)/LA.norm(y)
#z1=(z)/LA.norm(z)
#corr2 = signal.correlate2d(x1,y1)
#print(corr2)
#corr3 = signal.correlate2d(x1,z1)
#print(corr3)
#print(np.corrcoef(x,y))
#print(np.corrcoef(x,z))

#print(corr)