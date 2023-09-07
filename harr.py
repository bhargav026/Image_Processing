import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def analysisFilter(img):
    dim = img.shape
    xh_col = np.zeros([dim[0],dim[1]//2])
    xg_col = np.zeros([dim[0],dim[1]//2])
    xhh = np.zeros([dim[0]//2,dim[1]//2])
    xhg = np.zeros([dim[0]//2,dim[1]//2])
    xgh = np.zeros([dim[0]//2,dim[1]//2])
    xgg = np.zeros([dim[0]//2,dim[1]//2])
    d = np.sqrt(2)
    for i in range(dim[1]//2):
        xh_col[:,i] = (img[:,2*i]+img[:,2*i+1])/d
        xg_col[:,i] = (img[:,2*i]-img[:,2*i+1])/d

    for i in range(dim[1]//2):
        xhh[i,:] = (xh_col[2*i,:]+xh_col[2*i+1,:])/d
        xhg[i,:] = (xg_col[2*i,:]+xg_col[2*i+1,:])/d
        xgh[i,:] = (xh_col[2*i,:]-xh_col[2*i+1,:])/d
        xgg[i,:] = (xg_col[2*i,:]-xg_col[2*i+1,:])/d
    return (xhh,xhg,xgh,xgg)

def synthesisFilter(xhh,xhg,xgh,xgg):
    dim = xhh.shape
    xh_c = np.zeros([dim[0],dim[1]*2])
    xg_c = np.zeros([dim[0],dim[1]*2])
    x = np.zeros([2*dim[0],dim[1]*2])
    d = np.sqrt(2)
    xh_c[:,::2] = (xhh+xhg)/d
    xh_c[:,1::2] = (xhh-xhg)/d
    xg_c[:,::2] = (xgh+xgg)/d
    xg_c[:,1::2] = (xgh-xgg)/d

    x[::2,:] = (xh_c+xg_c)/d
    x[1::2,:] = (xh_c-xg_c)/d

    return x


img = cv2.imread('Lenna.jpg',cv2.IMREAD_GRAYSCALE)
(xhh,xhg,xgh,xgg) = analysisFilter(img.astype(int))
img_recon = synthesisFilter(xhh,xhg,xgh,xgg)



plt.figure()
plt.subplot(221)
plt.imshow(xhh,cmap='gray')
plt.subplot(222)
plt.imshow(xhg,cmap='gray')
plt.subplot(223)
plt.imshow(xgh,cmap='gray')
plt.subplot(224)
plt.imshow(xgg,cmap='gray')

plt.figure()
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.subplot(122)
plt.imshow(img_recon,cmap='gray')
plt.show()



