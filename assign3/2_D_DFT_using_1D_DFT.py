# -*- coding: utf-8 -*-
"""
Created on Thur Sept 05 11:31:56 2019

@author: Personal
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

a= cv2.imread('ImagesForAssignment3/zoneplate.png',cv2.IMREAD_GRAYSCALE)
M,N=a.shape
n= int(input('enter N value: '))

def dft(a,n):
    M,N=a.shape
    image=np.zeros((n,n),dtype=complex)
    def dft2(a):
        for col in range(0,a.shape[1],1):
            ndft[:,col]=np.fft.fft(a[:,col])
        for row in range(0,a.shape[0],1):
            image2[row,:]=np.fft.fft(ndft[row,:])
        return image2
    if(n==int(0.5*a.shape[0])):
        ndft=np.zeros((a.shape[0],a.shape[1]),dtype=complex)
        image2=np.zeros((a.shape[0],a.shape[1]),dtype=complex)
        image2=dft2(a)
        for i in range(n):
            for j in range(n):
                image[i][j]=image2[2*i][2*j]
    elif(n==2*a.shape[0] or n==a.shape[1]):
        ndft=np.zeros((n,n),dtype=complex)
        image2=np.zeros((n,n),dtype=complex)
        image3=np.zeros((n,n))
        if (n==2*a.shape[0]):
            image3[0:a.shape[0],0:a.shape[1]]=a
        elif(n==a.shape[0]):
            image3=a
        image=dft2(image3)
    return image

image=dft(a,n)
#dfft=np.fft.fft2(a)
image=np.fft.fftshift(image)
mag=20*np.log(np.abs(image))
phase=np.angle(image)
plt.figure()
plt.subplot(131),plt.imshow(a,cmap='gray')
plt.title('ZONE PLATE IMAGE'),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(mag,cmap='gray')
plt.title('2D-DFT MAGNITUDE'),plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(phase,cmap='gray')
plt.title('2D-DFT PHASE'),plt.xticks([]),plt.yticks([])
plt.show()
