# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:03:12 2019

@author: Personal
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider
import cv2 as cv
from PIL import Image
import scipy
from scipy import signal
from numpy import linalg as LA


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

a=cv.imread('cameraman.tif',cv.IMREAD_GRAYSCALE)
M,N=a.shape
mirror1=padding(a,2)
mirror=mirror1[M:M*M,N:M*N]

dft=DFT(mirror,mirror.shape[0])
#dft1=np.fft.fft2(mirror)
x,y=np.meshgrid(np.arange(0,M,1),np.arange(0,N,1))
DCTmatrix=np.exp(-((1j*np.pi*x)/2*M+(1j*np.pi*y)/2*N))
mydct=DCTmatrix*dft[0:M,0:N]
mydct=np.array(mydct, dtype='int64')

#normalsing
#mydct=(mydct-np.min(mydct))*255/np.max(mydct)
#mydct=(mydct-np.min(mydct))*255/np.max(mydct)

plt.figure()
plt.subplot(121)
plt.imshow(mirror, cmap='gray',vmin=0, vmax=255)
plt.title('INPUT IMAGE'),plt.xticks([]),plt.yticks([])

plt.subplot(122)
plt.imshow(mydct, cmap='gray')
plt.title('DCT')


indct=np.zeros([M,N])
for i in range(M):
    indct[i,:]=scipy.fftpack.dct(a[i,:])
for i in range(N):
    indct[:,i]=scipy.fftpack.dct(indct[:,i])


indct=np.array(indct, dtype='int64')


#indct=(mydct-np.min(indct))*255/np.max(indct)

diff=mydct-indct
plt.figure()
plt.subplot(121)
plt.imshow(indct,cmap='gray')
plt.title('INbuilt DCT')

plt.subplot(122)
plt.imshow(diff,cmap='gray')
plt.title('Differance')
