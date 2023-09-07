# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:12:31 2019

@author: Personal
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons,Slider
import datetime

start = datetime.datetime.now()
a=cv.imread('cameraman.tif', cv.IMREAD_GRAYSCALE)
plt.figure()
plt.subplot(121)
plt.imshow(a, cmap='gray', vmin=0, vmax=255)

def medianfilter(m,l):
    c=np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0],2*a.shape[0]):
        for j in range(a.shape[1],2*a.shape[1]):
            c[i-a.shape[0]][j-a.shape[1]]=np.median(m[i-int(l/2):i+int(l/2)+1,j-int(l/2):j+int(l/2)+1])
    return c  
b=np.pad(a,a.shape[0],'reflect')
o=medianfilter(b,11)    
stop_time = datetime.datetime.now()
plt.subplot(122)
out = plt.imshow(o, cmap='gray', vmin=0, vmax=255)
ax2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
L = Slider(ax2, 'L', 3,23,valinit=5, valstep=2)
def go2(label):
    
    l=int(label)
    b=np.pad(a,a.shape[0],'reflect')
    o=medianfilter(b,l)
    out.set_data(o)
    plt.draw()
    
L.on_changed(go2)