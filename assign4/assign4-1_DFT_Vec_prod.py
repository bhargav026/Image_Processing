# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:18:35 2019

@author: Personal
"""

#%matplotlib notebook
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons


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

d=cv2.imread('Lenna.jpg',cv2.IMREAD_GRAYSCALE)
print(d.shape)
img=dft(d,512)
img=idft(img,512)
plt.figure()
plt.subplot(121)
ip=plt.imshow(d, cmap='gray')
plt.title('LENNA IMAGE'),plt.xticks([]),plt.yticks([])
plt.subplot(122)
op=plt.imshow(img, cmap='gray')
plt.title('RECON.LENNA IMAGE'),plt.xticks([]),plt.yticks([])


radio=RadioButtons(plt.axes([0.01,0.75,0.17,0.25]),('1.P=N=M','2.P=N=2M','3.2P=N=M','4.P=N=M/2'),active=0)
def update(label):
    l=int(label[0],10)
    print(l)
    if(l==1):
        n=p=d.shape[0]
    elif(l==2):
        n=p=2*d.shape[0]
    elif(l==3):
        n=d.shape[0]
        p=int(n/2)
    elif(l==4):
        n=p=int(0.5*d.shape[0])
    e=dft(d,n)
    f=idft(e,p)
    op.set_data(f)
    plt.draw()
            
radio.on_clicked(update)
plt.show()