# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:58:59 2019

@author: Personal
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import cv2 as cv
from scipy.ndimage import gaussian_filter
import scipy
def linking(EM, thr1, thr2):
    p,q=EM.shape
    sm=EM*(EM>=thr1)
    wm=EM*(EM>=thr2)
    for k in range(5):
        for i in range(1,p-2):
            for j in range(1,q-2):
                try:
                    if(sm[i,j]!=0):
                        sm[i-2:i+3,j-2:j+3]=sm[i-2:i+3,j-2:j+3]-sm[i-2:i+3,j-2:j+3]+wm[i-2:i+3,j-2:j+3]
                            
                except IndexError as e:
                    pass
    sm=sm*255/np.max(sm)
    return sm


FIG=cv.imread('Lanes.jpg',0)
M,N=FIG.shape
f=np.array(gaussian_filter(FIG,1,mode='reflect'), dtype='float')
#f=np.array(cv.GaussianBlur(F,(3,9),5), dtype='float')

xfilter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
yfilter=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
dxf=np.array(scipy.ndimage.convolve(f,xfilter,mode='reflect'), dtype=float)
dyf=np.array(scipy.ndimage.convolve(f,yfilter,mode='reflect'), dtype=float)
tita=np.arctan2(dyf,dxf)*180/np.pi
modf2=np.array(np.sqrt(dxf**2+dyf**2),dtype=float)
modf=np.pad(modf2,1,'reflect')
modf1=np.zeros([M,N])
for i in range(1,M):
    for j in range(1,N):
        try:
            if(-22.5< tita[i,j] <= 22.5 or 157.5< tita[i,j]<=180 and -180<= tita[i,j] < -157.5):
                tita[i,j]=0
                a=modf[i,j+1]
                b=modf[i,j-1]
                
            elif(22.5< tita[i,j] <= 67.5 or -157.5<= tita[i,j]< -112.5):
                tita[i,j]=45
                a=modf[i-1,j+1]
                b=modf[i+1,j-1]
                
            elif(67.5< tita[i,j] <= 112.5 or -112.5<= tita[i,j]< -67.5):
                tita[i,j]=90
                a=modf[i-1][j]
                b=modf[i+1][j]
                
            elif(112.5< tita[i,j] <= 157.5 or -67.5<= tita[i,j]<= -22.5):
                tita[i,j]=135
                a=modf[i-1,j-1]
                b=modf[i+1,j+1]
                
            if (modf[i,j]>=a and modf[i,j]>=b):
                modf1[i,j]=modf[i,j]
            
        except IndexError as e:
            pass

modf2=modf1/np.max(modf1)
out=linking(modf2,0.25,0.15)
plt.figure()
plt.subplot(121)
plt.imshow(modf1, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('non maximal')
plt.subplot(122)
plt.imshow(out, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('final edge map')
ax1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
threshold1 = Slider(ax1, 'thr1', 0,1, valinit=0.25,valstep=0.01)
ax2= plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
threshold2 = Slider(ax2, 'thr2',0,1,valinit=0.15,slidermax=threshold1)
threshold1.slidermin=threshold2
def update(val):
    thr1=threshold1.val
    thr2=threshold2.val
    out=linking(modf2,thr1,thr2)
    plt.subplot(122)
    plt.imshow(out, cmap='gray')
    
threshold1.on_changed(update)
threshold2.on_changed(update)
