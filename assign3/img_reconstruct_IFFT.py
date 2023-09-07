# -*- coding: utf-8 -*-
"""
Created on Thur Sept 05 13:22:47 2019 

@author: Personal
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

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

b=cv2.imread('ImagesForAssignment3/Luna.png',cv2.IMREAD_GRAYSCALE)
M,N=b.shape
c=dft(b,512)
p=int(input('enter p value:'))
plt.figure()
def idft(ifft,p):
    ifft=np.conjugate(ifft)
    ifft=dft(ifft,p)
    if(p>M):
        ifft=np.conjugate(ifft)/(p*p)
    else:
        ifft=np.conjugate(ifft)/(M*M)
        
    ifft=ifft.real.astype(dtype=int)
    return ifft
c= idft(c,p)
if (p == M):
    diff=b-c
    plt.subplot(133),plt.imshow(diff, cmap='gray')
    plt.title('ERROR IMAGE'),plt.xticks([]),plt.yticks([])
    
plt.subplot(131),plt.imshow(b, cmap='gray')
plt.title('LUNA IMAGE'),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(c, cmap='gray')
plt.title('RECONSTRUCTED IMAGE'),plt.xticks([]),plt.yticks([])
plt.show()

