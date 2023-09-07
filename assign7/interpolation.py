import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

def bspline_bicubic(x):
    x = abs(x)
    if 0 <= x < 1:
        return (2/3) - 0.5*(x*x)*(2 - x)
    elif 1<= x < 2:
        return (1/6)*((2-x)**3)
    else:
        return 0

def bspline_linear(x):
    if -1 <= x < 0:
        return 1 + x
    elif 0 <= x <= 1:
        return 1 - x
    else:
        return 0

def interpolator(i, j, axis, technique):
    if axis == 1:
        value = 0
        for l in range(0, N):
            value = value + img[i, l] * technique(j-l)
        return value
    elif axis == 0:
        value = 0
        for l in range(0, M):
            value = value + dummy_row_img[l, j] * technique(i-l)
        return value

def interpolation_of_image(img, upsampling_factor, technique):
    for i in range(0, M):
        for j in range(0, k*N):
            dummy_row_img[i, j] = interpolator(i, j/k, 1, technique)
        
    for j in range(0, k*N):
        for i in range(0, k*M):
            dummy_col_img[i, j] = interpolator(i/k, j, 0, technique)
    
    return dummy_col_img

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.35)

img=cv.imread('cameraman.tif',0)
img = cv.resize(img, (64, 64))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Input image')
    
k = 2
M, N = img.shape

dummy_row_img = np.zeros((M, k*N))
dummy_col_img = np.zeros((k*M, k*N))
linearly_interpolated_image = interpolation_of_image(img, k, bspline_linear)

dummy_row_img = np.zeros((M, k*N))
dummy_col_img = np.zeros((k*M, k*N))
bicubic_interpolated_image = interpolation_of_image(img, k, bspline_bicubic)

plt.subplot(132)
plt.imshow(linearly_interpolated_image, cmap='gray')
plt.title('Linear Interpolation')

plt.subplot(133)
plt.imshow(bicubic_interpolated_image, cmap='gray')
plt.title('Bicubic Interpolation')
