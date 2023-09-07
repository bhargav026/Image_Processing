# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:11:32 2019

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

img=cv.imread('Lenna.jpg',0)
#M,N=img1.shape
#img1=np.array([[1,4,-3,0],[1,1,1,1]])
def convolve(in1,in2):
    in3=np.zeros([in1.shape[0],in1.shape[1]+1])
    for i in range(0,in1.shape[0],1):
        in3[i,:]=np.convolve(in1[i,:],in2)
    return in3[:,0:in1.shape[1]]
def filter_bank(img1,n):
    if n==0:
        return img1
    else:
        A=np.zeros_like(img1)
        lpf=np.array([1/np.sqrt(2),1/np.sqrt(2)])
        hpf=np.array([1/np.sqrt(2),-1/np.sqrt(2)])
        Lout=convolve(img1,lpf)
        Hout=convolve(img1,hpf)
        Lout=((Lout-np.min(Lout))/np.max(Lout))*255
        Hout=((Hout-np.min(Hout))/np.max(Hout))*255
        Lout1=Lout[::,::2]
        LL=convolve(Lout1.T,lpf)
        LL1=LL[::,::2].T
        LH=convolve(Lout[::,::2].T,hpf)
        HL=convolve(Hout[::,::2].T,lpf)
        HH=convolve(Hout[::,::2].T,hpf)
        LL1=((LL1-np.min(LL1))/np.max(LL1))*255
        HL=((HL-np.min(HL))/np.max(HL))*255
        HH=((HH-np.min(HH))/np.max(HH))*255
        LH=((LH-np.min(LH))/np.max(LH))*255
        A[img1.shape[0]//2:img1.shape[0],0:img1.shape[1]//2]=HL[::,::2].T
        A[img1.shape[0]//2:img1.shape[0],img1.shape[1]//2:img1.shape[1]]=HH[::,::2].T
        A[0:img1.shape[0]//2,img1.shape[1]//2:img1.shape[1]]=LH[::,::2].T
        A[0:img1.shape[0]//2,0:img1.shape[1]//2]=filter_bank(LL1,n-1)
    return A
    
def inversefilter_bank(img2,n):
    for i in range(1,n+2,1):
        
        if i==n+1:
            return img2
        else:
            img1=img2[0:img2.shape[0]//2**(n-i),0:img2.shape[1]//2**(n-i)]
            B=np.zeros_like(img1)
            ALL=img1[0:img1.shape[0]//2,0:img1.shape[1]//2]
            AHH=img1[img1.shape[0]//2:img1.shape[0],img1.shape[1]//2:img1.shape[1]]
            AHL=img1[img1.shape[0]//2:img1.shape[0],0:img1.shape[1]//2]
            ALH=img1[0:img1.shape[0]//2,img1.shape[1]//2:img1.shape[1]]
            UALL=np.zeros([img1.shape[1]//2,img1.shape[0]])
            UALH=np.zeros([img1.shape[1]//2,img1.shape[0]])
            UAHL=np.zeros([img1.shape[1]//2,img1.shape[0]])
            UAHH=np.zeros([img1.shape[1]//2,img1.shape[0]])
            
            UALL[::,::2]=ALL
            UALH[::,::2]=ALH
            UAHL[::,::2]=AHL
            UAHH[::,::2]=AHH
            
            ilpf=np.array([1/np.sqrt(2),1/np.sqrt(2)])
            ihpf=np.array([-1/np.sqrt(2),1/np.sqrt(2)])
            
            LL=convolve(UALL,ilpf)
            LL=((LL-np.min(LL))/np.max(LL))*255
            LH=convolve(UALH,ihpf)
            LH=((LH-np.min(LH))/np.max(LH))*255
            HL=convolve(UAHL,ilpf)
            HL=((HL-np.min(HL))/np.max(HL))*255
            HH=convolve(UAHH,ihpf)
            HH=((HH-np.min(HH))/np.max(HH))*255
            iLout=(LL+LH).T
            iHout=(HH+HL).T
            
            ULout=np.zeros([img1.shape[0],img1.shape[1]])
            UHout=np.zeros([img1.shape[0],img1.shape[1]])
            
            ULout[::,::2]=iLout
            UHout[::,::2]=iHout
            
            L=convolve(ULout,ilpf)
            L=((L-np.min(L))/np.max(L))*255
            H=convolve(UHout,ihpf)
            H=((H-np.min(H))/np.max(H))*255
            img2[0:img2.shape[0]//2**(n-i),0:img2.shape[1]//2**(n-i)]=(L+H).T
            
n = 3
out=filter_bank(img,n)
out2=out.copy()
out1=inversefilter_bank(out2,n)
#n=2
#for i in range(0,n):
#    Hsize=int(N/2**i)
#    img1=np.copy(output[0:Hsize,0:Hsize])
#    output[0:Hsize,0:Hsize]=filter_bank(Hsize,img1)

plt.figure()
plt.subplot(121)
plt.imshow(np.float64(out), cmap='gray')
plt.subplot(122)
plt.imshow(np.float64(out1), cmap='gray')
