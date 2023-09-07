import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

#load the image
img = mpimg.imread('lena_gray_512.tif')
plt.figure()
imgplot = plt.imshow(img, cmap ='gray', vmin = 0, vmax = 255)
plt.show()

#set the thresholds
thr1 = 100
thr2 = 200

#aaply them on filter
out_img = img.copy()
out_img[out_img<thr1] = thr1
out_img[out_img>thr2]= thr2


fig = plt.figure()
ax = plt.axes()
input_char = np.arange(0, 256)
output_char = np.arange(0, 256)
output_char[input_char < thr1] = thr1
output_char[input_char > thr2] = thr2
img_char,= ax.plot(input_char, input_char)
img_char.set_ydata(output_char)


plt.figure()
imgplot1 = plt.imshow(out_img, cmap ='gray', vmin = 0, vmax = 255)
plt.show()

ax1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='b')
ax2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='b')
thr1_control = Slider(ax1, 'threshold1', 0, 255, valinit=thr1, valstep=1)
thr2_control = Slider(ax2, 'threshold2', 0, 255, valinit=thr2, valstep=1, slidermin=thr1_control)
thr1_control.slidermax = thr2_control

def update(val):
    thr1 = thr1_control.val
    thr2 = thr2_control.val
    
    out_img = img.copy()
    out_img[out_img < thr1] = thr1
    out_img[out_img > thr2] = thr2
    output_char = np.arange(0, 256)
    output_char[input_char < thr1] = thr1
    output_char[input_char > thr2] = thr2
    # set the result in the plot
    imgplot1.set_data(out_img)
    img_char.set_ydata(output_char)
    fig.canvas.draw()

thr1_control.on_changed(update)
thr2_control.on_changed(update)