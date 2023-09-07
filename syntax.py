# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:19:16 2019

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
from mpl_toolkits.mplot3d import axes3d

#to get plots in seperate windows  
  %matplotlib qt
  
# to comeback to original state
  %matplotlib inline
  
#to check wheather input is given format or not
    if (isinstance(n, int) and n >=0):
        
#if condition can be write like this also        
    if( n is (0 or 1)):
# in list except 1st and last n elements  it takes
    my_list[n:-n]

[i^2 for i in range(0,4)] gives [0,1,4,9]

 for color= xkcd:colourname # in xckd we get 954 colours or  we can give hexa code
             CSS:
#reading image
b=cv2.imread('Luna.png',cv2.IMREAD_GRAYSCALE)

# indexing
out_img = img.copy()
out_img[out_img < thr1] = thr1
out_img[out_img > thr2] = thr2
#we can use like this also
HCI=1*(HCI>3000)

#sliders
ax1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='b')
ax2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='b')
thr1_control = Slider(ax1, 'Threshold1', 0, 255, valinit=0, valstep=1)
thr2_control = Slider(ax2, 'Threshold2', 0, 255, valinit=0, valstep=1)
def update()
thr1_control.on_changed(update)

#radiobutton
radio=RadioButtons(plt.axes([0.01,0.75,0.17,0.25]),('1.P=N=M','2.P=N=2M','3.2P=N=M','4.P=N=M/2'),active=0)
radio.on_clicked(update)

#mesh grid and 3D plot
ax = plt.axes(projection='3d')
i=np.arange(0, 50,1)
j=np.arange(0,50,1)  
xx, yy= np.meshgrid(i,j)
ax.plot_surface(xx,yy,img2)

