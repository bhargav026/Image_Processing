# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:34:16 2019

@author: SPECTRUM
"""

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


img = mpimg.imread('lena_gray_512.tif')
plt.figure()
imgplot = plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
plt.show()

t1 = 100
t2 = 200

fig = plt.figure()
ax = plt.axes()
input_char = np.arange(0, 256)
output_char = np.arange(0, 256)
output_char[input_char < t1] = t1
output_char[input_char > t2] = t2
img_char, = ax.plot(input_char, input_char)
img_char.set_ydata(output_char)

modified_img = img.copy()
modified_img[img < t1] = t1
modified_img[img > t2] = t2
plt.figure()
imgplot2 = plt.imshow(modified_img, cmap='gray', vmin=0, vmax=255)
plt.show()

plt.subplots_adjust(left=0.10, bottom=0.25)

axcolor = 'lightgoldenrodyellow'
t1_axes = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
t2_axes = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
t1_vals = Slider(t1_axes, 't1', 1, 254, valinit = t1)
t2_vals = Slider(t2_axes, 't2', 1, 254, valinit = t2, slidermin = t1_vals)
t1_vals.slidermax = t2_vals

def update(val):
    t1 = t1_vals.val
    t2 = t2_vals.val
    modified_img=img.copy()
    modified_img[img < t1] = t1
    modified_img[img > t2] = t2
    output_char = np.arange(0, 256)
    output_char[input_char < t1] = t1
    output_char[input_char > t2] = t2
    imgplot2.set_data(modified_img)
    img_char.set_ydata(output_char)
    fig.canvas.draw()

t1_vals.on_changed(update)
t2_vals.on_changed(update)
