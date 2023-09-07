# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:55:20 2019

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
from scipy import ndimage
from numpy import linalg as LA
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.35)

rax = plt.axes([0.05, 0.1, 0.15, 0.15],facecolor= 'lightgoldenrodyellow')
radio = RadioButtons(rax, ('Zero padding', 'Mirror','Periodic'))
ax1=plt.axes([0.25, 0.15, 0.5, 0.02], facecolor='b')
Variance_of_guassian = Slider(ax1,'Variance_of_guassian_fil',  0.1, 30, valinit=0.1, valstep=0.5)
ax2=plt.axes([0.25, 0.05, 0.5, 0.02], facecolor='b')
PSNR_control = Slider(ax2,'PSNR', 10, 30, valinit=10, valstep=1)

img1=cv.imread('cameraman.tif', 0)
plt.subplot(131)
plt.imshow(img1, cmap='gray', vmin =0, vmax =255 )
plt.show()
plt.title('Input image')
plt.xticks([])
plt.yticks([])
M=img1.shape[0]
img=img1.copy()

def Pad_zeros(img,size):
    img_dummy=np.zeros([3*size,3*size])
    img_copy=img.copy()
    img_dummy[size:2*size,size:2*size]=img_copy
    return img_dummy

def Periodic_bound(img,size):
    img_dummy=np.zeros([3*size,3*size])
    img_copy=img.copy()
    for k in range(0,3):
        for l in range(0,3):
            img_dummy[k*size:(k*size)+size,l*size:(l*size)+size]=img_copy
    return img_dummy

def Mirror_bound(img,size):
    img_dummy=np.zeros([3*size,3*size])
    img_copy=img.copy()
    for k in range(0,3):
        for l in range(0,3):
            img_temp=np.zeros([size,size])
            if (k==0 and not(l==1)) or (k==2 and not(l==1)):
                img_temp=np.flip(np.flip(img_copy,1),0)
            elif (k==1 and not(l==1)):
                img_temp=np.flip(img_copy,1)
            elif (l==1 and not(k==1)):
                img_temp=np.flip(img_copy,0)
            else:
                img_temp=img_copy
            img_dummy[k*size:(k*size)+size:,l*size:(l*size)+size:]=img_temp
           
    return img_dummy


def Moving_avg_filt(img_dummy1,L,M):
    img_dummy=img_dummy1.copy()
    img_dummy_row=np.zeros([M,M+2*L])
    img_dummy_col=np.zeros([M,M])
    img_dummy1=img_dummy[M-L:2*M+L,M-L:2*M+L]
    img_mask=img_dummy1[0:(2*L)+1,:]
    img_dummy_row[0,:]=np.sum(img_mask,axis=0)/((2*L)+1)
    for i in range(1,M-1):
        img_dummy_row[i,:]=img_dummy_row[i-1,:]+(img_dummy1[i+2*L+1,:]-img_dummy1[i-1,:])/((2*L)+1)
   
    img_mask1=img_dummy_row[:,0:(2*L)+1]
    img_dummy_col[:,0]=np.sum(img_mask1,axis=1)/((2*L)+1)
    for i in range(1,M-1):
        img_dummy_col[:,i]=img_dummy_col[:,i-1]+(img_dummy_row[:,i+2*L+1]-img_dummy_row[:,i-1])/((2*L)+1)
    return img_dummy_col
 
def Create_img_PSNR(img,PSNR):
    img1=img.copy()
    std_dev=np.max(img1)/(10**(PSNR/20))
    s = np.random.normal(0,std_dev, (img1.shape[0],img1.shape[1]))
    return img1+s;

def PSNR_gain_cal(Input_image,Noise_image,reconstruct_image):
    IPSNR=20*np.log10(255/(np.std(Input_image-Noise_image)))
    OPSNR=20*np.log10(255/(np.std(Input_image-reconstruct_image)))
    return OPSNR-IPSNR
   
   
def update(val):
    B_Condition=radio.value_selected
    variance=Variance_of_guassian.val
    PSNR=PSNR_control.val
    img=Create_img_PSNR(img1,PSNR)
    img2=img.copy()
    L=3
    N=int(np.ceil(variance/4))
   
    for i in range(0,N):
        img_dummy=np.zeros([3*M,3*M])
        if B_Condition=='Zero padding':
            img_dummy=Pad_zeros(img,img.shape[0])
            plt.subplot(132)
            plt.imshow(img_dummy,cmap='gray')
            plt.title('Zero padding')

        elif B_Condition=='Periodic':
            img_dummy=Periodic_bound(img,img.shape[0])
            plt.subplot(132)
            plt.imshow(img_dummy,cmap='gray')
            plt.title('Periodic')                
   
        else:
            img_dummy=Mirror_bound(img,img.shape[0])        
            plt.subplot(132)
            plt.imshow(img_dummy,cmap='gray')
            plt.title('Mirror')
       

        Mov_Avg_output=Moving_avg_filt(img_dummy,L,M)
        img=Mov_Avg_output.copy()
   
    #PSNR_GAIN=PSNR_gain_cal(img1,img2,img)
    plt.subplot(133)
    plt.imshow(np.float64(img),cmap='gray')
    plt.title('Filtered image')

Variance_of_guassian .on_changed(update)
PSNR_control.on_changed(update)
radio.on_clicked(update)