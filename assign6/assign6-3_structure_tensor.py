# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:40:58 2019

@author: Personal
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from scipy.ndimage import gaussian_filter
import scipy

img=cv.imread('Fingerprint.jpg',0)
M,N=img.shape
f_img=np.array(gaussian_filter(img,1,mode='reflect'), dtype='float')
xfilter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
yfilter=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
dxf=np.array(scipy.ndimage.convolve(f_img,xfilter,mode='reflect'), dtype=float)
dyf=np.array(scipy.ndimage.convolve(f_img,yfilter,mode='reflect'), dtype=float)
dxf2=dxf*dxf
dyf2=dyf*dyf
dxyf=dxf*dyf
j11=np.array(gaussian_filter(dxf2,1,mode='reflect'), dtype='float')
j22=np.array(gaussian_filter(dyf2,1,mode='reflect'), dtype='float')
j12=np.array(gaussian_filter(dxyf,1,mode='reflect'), dtype='float')

energy=j11+j22

coherance=np.sqrt(energy*energy - 4*(j11*j22-j12*j12))/energy
#coherance=np.nan_to_num(coherance, 1.0)
orientation=np.arctan2(2*j12,j11-j22)*90/np.pi

#orientation=np.nan_to_num(orientation,1)
print(np.max(orientation))
print(np.min(orientation))

plt.figure()
plt.subplot(121)
plt.imshow(energy, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(coherance, cmap='gray')

plt.figure()
plt.imshow(orientation, cmap='hsv')
plt.colorbar()
