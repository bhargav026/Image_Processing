# -*- coding: utf-8 -*-
"""
Created on Thur Sept 05 14:38:04 2019

@author: Personal
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons


def dft(a,n):

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

def idft(ifft,p):
    ifft=np.conjugate(ifft)
    ifft=dft(ifft,p)
    if(p>M):
        ifft=np.conjugate(ifft)/(p*p)
    else:
        ifft=np.conjugate(ifft)/(M*M)
        
    ifft=ifft.real.astype(dtype=int)
    return ifft


d=cv2.imread('ImagesForAssignment3/Lenna.jpg',cv2.IMREAD_GRAYSCALE)
M,N=d.shape
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