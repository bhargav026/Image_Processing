# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:07:15 2019

@author: Personal
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons,Slider
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
input_image= cv.imread('cameraman.tif',cv.IMREAD_GRAYSCALE)
l=3
output_image=np.zeros((768,768))
output_image=padding(input_image,l)
#print(output_image)

plt.figure()
output,_,_=plt.imshow(output_image, cmap='gray',vmin=0, vmax=255), plt.xticks([]),plt.yticks([])
pad=RadioButtons(plt.axes([0.005,0.5,0.17,0.25]),('1.zero','2.mirror','3.periodic'),active=1)

def update(label):
    
    l=int(label[0],10)
    output_image=padding(input_image,l)
    output_image=np.array(output_image)
    output.set_data(output_image)
    plt.draw()
    
pad.on_clicked(update)
plt.show()