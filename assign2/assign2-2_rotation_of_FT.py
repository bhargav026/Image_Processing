# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:12:10 2019

@author: Personal
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

mm,nn=np.meshgrid(range(256),range(256))
def cosinwave(w,t):
    
    c=np.cos(w*(mm*np.cos(t)+nn*np.sin(t)))
    dft=np.fft.fft2(c)
    fshift = np.fft.fftshift(dft)
    ms =20*np.log(np.abs(fshift))
    ps = np.angle(fshift)
    return(c,ms,ps)
    
c,ms,ps=cosinwave(0.9,0.6)
plt.subplot(131)
plane=plt.imshow(c,cmap='gray')
plt.title('A PLANE'),plt.xticks([]), plt.yticks([])
plt.subplot(132)
mag=plt.imshow(ms, cmap='gray')
plt.title('FOURIER MAGNITUDE'),plt.xticks([]), plt.yticks([])
plt.subplot(133)
phase=plt.imshow(ps, cmap='gray')
plt.title('FOURIER PHASE'),plt.xticks([]), plt.yticks([])

ax2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax3 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
w_control = Slider(ax2, 'w', 0, np.pi, valinit=0, valstep=0.02)
tita_control = Slider(ax3, 'tita', 0, 2*np.pi, valinit=0, valstep=0.04)

def update(val):
    
    w= w_control.val
    t = tita_control.val
    c,ms,ps=cosinwave(w,t)
    
    plane.set_data(c)    
    mag.set_data(ms)
    phase.set_data(ps)
    plt.draw()
    
w_control.on_changed(update)
tita_control.on_changed(update)
plt.show()