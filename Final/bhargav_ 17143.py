# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:06:11 2019

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


def DFT(a,n):
    x,y=np.meshgrid(np.arange(0,a.shape[0],1),np.arange(0,n,1))
    dftm=np.exp(-((1j*np.pi*2*x*y)/n))
    mm=np.zeros((n,a.shape[0]),dtype=complex)
    nn=np.zeros((a.shape[1],n),dtype=complex)
    image2=np.zeros((n,n),dtype=complex)
    mm=np.matmul(dftm,a)
    nn=np.transpose(mm)
    image2=np.matmul(dftm,nn)
    image2=np.transpose(image2)
    return image2
def idft(b,p):
    ifft=np.conjugate(b)
    ifft=DFT(ifft,p)
    ifft=np.conjugate(ifft)/(p*p)
    ifft=ifft.real.astype(dtype=int)
    return ifft


a=cv.imread('pirate.tif',cv.IMREAD_GRAYSCALE)
H4=np.array([[0,14,3,13],[11,5,8,6],[12,2,15,1],[7,9,4,10]])
H2=np.array([[0,2],[3,1]])
N= int(input('enter type: 1 for H4,  2 for H2 '))
DFT_img=DFT(a,2048)

DFT_H4=DFT(H4,4)
DFT_H2=DFT(H2,2)
if(N==1):
    Filter=DFT_H4
elif(N==2):
    Filter=DFT_H2

dummy=np.zeros([int(DFT_img.shape[0]/Filter.shape[0]),int(DFT_img.shape[1]/Filter.shape[1])])
for i in range(0,DFT_img.shape[0],Filter.shape[0]):
    for j in range(0,DFT_img.shape[1],Filter.shape[1]):
        dummy[int(i/Filter.shape[0]),int(j/Filter.shape[1])]=np.sum(DFT_img[i:i+Filter.shape[0],j:j+Filter.shape[1]]*Filter)
        

Recon_img=idft(dummy,512)
plt.figure()
plt.subplot(121)
ip=plt.imshow(a, cmap='gray')
plt.title('PIRATE IMAGE')#,plt.xticks([]),plt.yticks([])
plt.subplot(122)
op=plt.imshow(Recon_img, cmap='gray')
plt.title('halftoned image'),plt.xticks([]),plt.yticks([])

Recon_img=np.array(Recon_img*255/np.max(Recon_img))
c=Recon_img+np.random.normal(0,np.sqrt(10),(512,512))
out_img =c.copy()
out_img[out_img > 255] = 255
out_img[out_img < 0] = 0
out_img[out_img < 127] = 0
out_img[out_img > 127] = 1

plt.figure()
plt.subplot(121)
plt.imshow(out_img, cmap='gray')
plt.title('one bit qunatised')#,plt.xticks([]),plt.yticks([])
out_img[out_img ==0] = 64
out_img[out_img ==1] = 192
plt.subplot(122)
plt.imshow(out_img, cmap='gray', vmin=0,vmax=255)
plt.title('quatization as Gray')
