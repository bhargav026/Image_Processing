import matplotlib.image as mpimg
from matplotlib import pyplot as plt

img1 = mpimg.imread('pirate.tif')
img2 = mpimg.imread('lena_gray_512.tif')

plt.subplot(131)
plt.imshow(img1, cmap= 'gray', vmin =0, vmax=255)
plt.subplot(132)
plt.imshow(img2, cmap= 'gray', vmin =0, vmax=255)

out_img = img1.copy()
out_img[img1<img2] = 0
out_img[img1>=img2] = 255

plt.subplot(133)
plt.imshow(out_img, cmap= 'gray', vmin =0, vmax=255)
plt.show()
