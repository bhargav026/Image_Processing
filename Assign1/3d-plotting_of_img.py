import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np



img = mpimg.imread('cameraman.tif')

plt.figure()
plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
plt.show()
x,y = np.meshgrid(np.arange(img.shape[0]),np.arange( img.shape[1]))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x,y, img, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('intensity');