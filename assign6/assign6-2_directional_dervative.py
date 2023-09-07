# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:24:45 2019

@author: Personal
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from cv2 import VideoWriter, VideoWriter_fourcc

width = 1200
height = 600
FPS = 24
seconds = 15
t=45
F=cv.imread('PCB.jpg',0)
f=np.array(gaussian_filter(F,5,mode='reflect'), dtype='float')
#f=np.array(cv.GaussianBlur(F,(3,9),5), dtype='float')

xfilter=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
yfilter=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
dxf=np.array(ndimage.convolve(f,xfilter,mode='reflect'), dtype=np.int16)
dyf=np.array(ndimage.convolve(f,yfilter,mode='reflect'), dtype=np.int16)
dtf=dxf*np.cos(t*np.pi/180)+dyf*np.sin(t*np.pi/180)
plt.figure()
plt.subplot(121)
plt.imshow(F, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.title('PCB')
plt.subplot(122)
plt.imshow(dtf, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.title('smoothed PCB')


fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./derivative3.avi', fourcc, float(FPS), (width, height), isColor=False)

for t in range(360,0,-1):
    T=t*2*np.pi/360
    dtf=np.array(dxf*np.cos(T)+dyf*np.sin(T), dtype= np.float32)
    dtf=dtf-np.min(dtf)
    dtf1=np.array(dtf,dtype=np.uint8)
    video.write(dtf1)
video.release()
#for t in range(FPS*seconds):
#    T=t*2*np.pi/360
#    dtf=dxf*np.cos(T)+dyf*np.sin(T)
    
    

