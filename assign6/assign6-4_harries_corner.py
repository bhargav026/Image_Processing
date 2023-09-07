# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:26:53 2019

@author: Personal
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from scipy.ndimage import gaussian_filter
import scipy
img=cv.imread('PCB.jpg',0)
img1=cv.imread('PCB.jpg')
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
HCI=j11*j22-j12*j12-0.1*energy**2
HCI=1*(HCI>7000)

for i in range(0,M):
    for j in range(0,N):
        if(HCI[i,j]==1):
            img1[i,j,:]=[255,0,0]
            
plt.figure()
plt.imshow(img1)

