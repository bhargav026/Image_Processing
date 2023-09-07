# -*- coding: utf-8 -*-
"""
Created on Thur Sept 05 16:44:22 2019

@author: Personal
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
import matplotlib.image as mpimg

def dft(a,n):
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
    ifft=dft(ifft,p)/(p*p)
    ifft=np.conjugate(ifft)
    ifft=ifft.real.astype(dtype=int)
    return ifft

img1 = cv2.imread("ImagesForAssignment3/Rajinikanth.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("ImagesForAssignment3/Amitabh.jpg", cv2.IMREAD_GRAYSCALE)

dft_img1 = dft(img1, img1.shape[0])
dft_img2 = dft(img2, img2.shape[0])


mag1=np.abs(dft_img1)
phase1=np.angle(dft_img1)

mag2=np.abs(dft_img2)
phase2=np.angle(dft_img2)

#case1
f3 = idft(mag1*np.exp(1j*phase2), mag1.shape[0])

#case2
f4 = idft(mag2*np.exp(1j*phase1), mag2.shape[0])

plt.subplot(221)
plt.imshow(img1, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("f1_Rajinikanth"),plt.xticks([]),plt.yticks([])

plt.subplot(222)
plt.imshow(img2, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("f2_Amitabh"),plt.xticks([]),plt.yticks([])

plt.subplot(223)
plt.imshow(f3, cmap ='gray', vmin= 0, vmax= 255)
plt.title("f3_Case1"),plt.xticks([]),plt.yticks([])

plt.subplot(224)
plt.imshow(f4, cmap ='gray', vmin= 0, vmax= 255)
plt.title("f4_Case2"),plt.xticks([]),plt.yticks([])