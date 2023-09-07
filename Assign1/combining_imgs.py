import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

img1 = mpimg.imread('pirate.tif')
img2 = mpimg.imread('lena_gray_512.tif')

plt.subplot(121)
plt.imshow(img1, cmap= 'gray', vmin =0, vmax=255)
plt.subplot(122)
plt.imshow(img2, cmap= 'gray', vmin =0, vmax=255)
plt.show()

a = 0.75

out_img = np.ceil(a*img1+(1-a)*img2)
min = np.min(out_img)
max = np.max(out_img)
out_img = (out_img-min)/(max-min)
out_img= out_img*255

plt.figure()
imgplot2 = plt.imshow(out_img, cmap='gray', vmin=0, vmax=255)
plt.show()

a_axes = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='b')
a_vals = Slider(a_axes, 'a', 0,1, valinit = a)

def update(val):
    
    a =a_vals.val
    out_img = np.ceil(a*img1+(1-a)*img2)
    min = np.min(out_img)
    max = np.max(out_img)
    out_img = (out_img-min)/(max-min)
    out_img= out_img*255
    imgplot2.set_data(out_img)

a_vals.on_changed(update)

