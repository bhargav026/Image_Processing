# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:44:46 2019

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
from scipy import ndimage
from cv2 import VideoWriter, VideoWriter_fourcc


F=cv.imread('PCB.jpg',0)
#f=gaussian_filter(F,2,mode='reflect')
f=cv.GaussianBlur(F,(1,5),3)
f1=f[123:127,0:4]
xfilter=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
yfilter=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
dxf=np.array(ndimage.convolve(f1,xfilter,mode='reflect'), dtype=np.int32)
dyf=np.array(ndimage.convolve(f1,yfilter,mode='reflect'), dtype=np.int32)