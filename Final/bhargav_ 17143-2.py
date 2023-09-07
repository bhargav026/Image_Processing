# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:06:51 2019

@author: Personal
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
from scipy.ndimage import gaussian_filter
import scipy


def padding(n,t):
    out=np.zeros((3*n.shape[0],3*n.shape[1]))
    if(t==1):
        out[n.shape[0]:2*n.shape[0],n.shape[1]:2*n.shape[1]]=n
    
    elif(t==2):
        n1=np.fliplr(n)
        m=np.concatenate((n1,n,n1), axis=1)
        m1=np.flipud(m)
        out=np.concatenate((m1,m,m1), axis=0)
        
    elif(t==3):
        for i in range(0,3*n.shape[1],n.shape[1]):
            for j in range(0,3*n.shape[0],n.shape[0]):
                out[j:j+256,i:i+256]=n
    return out   

def iteavgfilter(n,NN):
    n=np.array(n, dtype='float')
    for k in range(0,NN,1):
        bb=np.zeros((int(n.shape[0]/3),int(n.shape[1]/3)), dtype='float')
        for i in range(int(n.shape[0]/3), 2*int(n.shape[0]/3),1):
            n[i][int(n.shape[1]/3)]=np.sum(n[i-L:i+L+1,int(n.shape[1]/3)-L:int(n.shape[1]/3)+L+1])/(2*L+1)**2
            for j in range(int(n.shape[1]/3)+1,2*int(n.shape[1]/3),1):
                n[i][j]=n[i][j-1]+(np.sum(n[i-L:i+L+1,j+L:j+L+1])-np.sum(n[i-L:i+L+1,j-L:j-L+1]))/(2*L+1)**2
        bb=n[int(n.shape[0]/3):2*int(n.shape[0]/3),int(n.shape[1]/3):2*int(n.shape[1]/3)]
        n=padding(bb,2)
    #print(bb)
    return bb


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

b=cv.imread('RiceGrains.tif', cv.IMREAD_GRAYSCALE)
M,N=b.shape
L=3
pad_img=padding(b,2)
filterd_img=iteavgfilter(pad_img,2)

plt.figure()
plt.subplot(121)
ip=plt.imshow(b, cmap='gray')
plt.title('PIRATE IMAGE')#,plt.xticks([]),plt.yticks([])
plt.subplot(122)
op=plt.imshow(filterd_img, cmap='gray')
plt.title('filterd_img'),plt.xticks([]),plt.yticks([])

xfilter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
yfilter=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
dxf=np.array(scipy.ndimage.convolve(filterd_img,xfilter,mode='reflect'), dtype=float)
dyf=np.array(scipy.ndimage.convolve(filterd_img,yfilter,mode='reflect'), dtype=float)
tita=np.arctan2(dyf,dxf)*180/np.pi
modf2=np.array(np.sqrt(dxf**2+dyf**2),dtype=float)
modf3=padding(modf2,2)
modf=modf3[modf2.shape[0]-1:2*modf2.shape[0]+1,modf2.shape[1]-1:2*modf2.shape[1]+1]
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
out=linking(modf2,0.25,0.20)
plt.figure()
plt.subplot(121)
plt.imshow(out, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('boundary map')
out1=np.array(out/np.max(out), dtype=float)
dummy1=255*(out1>=0.3)
dummy3=padding(dummy1,1)
p=17


dummy2=dummy3[dummy1.shape[0]-p:2*dummy1.shape[0]+p,dummy1.shape[1]-p:2*dummy1.shape[1]+p]
for i in range(p,dummy1.shape[0]+p):
    for j in range(p,dummy1.shape[1]+p):
        if dummy2[i,j]==255:
            dummy2[i-p:i+p,j-p:j+p]=0
            dummy2[i,j]=255
plt.subplot(122)
plt.imshow(dummy2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('location of grains')

#plt.figure()     
#plt.subplot(121)
#plt.imshow(out, cmap='gray')
#plt.xticks([]), plt.yticks([])
#plt.title('location of grains')



