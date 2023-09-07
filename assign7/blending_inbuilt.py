# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:14:18 2019

@author: Madhu
"""
#import cv2 as cv
#import numpy as np
#
#def Mirror_bound(img):
#    M, N = img.shape
#    img_dummy = np.zeros([3*M, 3*N])
#    img_copy = img.copy()
#    for k in range(0, 3):
#        for l in range(0, 3):
#            img_temp = np.zeros([M, N])
#            if (k == 0 and not(l == 1)) or (k == 2 and not(l == 1)):
#                img_temp = np.flip(np.flip(img_copy, 1), 0)
#            elif (k == 1 and not(l == 1)):
#                img_temp = np.flip(img_copy, 1)
#            elif (l == 1 and not(k == 1)):
#                img_temp = np.flip(img_copy, 0)
#            else:
#                img_temp = img_copy
#            img_dummy[k*M:(k*M)+M:, l*N:(l*N)+N:] = img_temp 
#            
#    return img_dummy
#
#def Moving_avg_filt(img_dummy1, L, M, N):
#    img_dummy = img_dummy1.copy()
#    img_dummy_row = np.zeros([M, N+2*L])
#    img_dummy_col = np.zeros([M, N])
#    img_dummy1 = img_dummy[M-L:2*M+L, N-L:2*N+L]
#
#    img_mask = img_dummy1[0:(2*L)+1, :]
#    img_dummy_row[0, :] = np.sum(img_mask, axis = 0)/((2*L)+1)
#    
#    for i in range(1, M-1):
#        img_dummy_row[i, :] = img_dummy_row[i-1, :]+(img_dummy1[i+2*L+1, :]-img_dummy1[i-1, :])/((2*L)+1)
#   
#    img_mask1 = img_dummy_row[:, 0:(2*L)+1]
#    img_dummy_col[:, 0] = np.sum(img_mask1, axis = 1)/((2*L)+1)
#    for i in range(1, N-1):
#        img_dummy_col[:, i] = img_dummy_col[:, i-1]+(img_dummy_row[:, i+2*L+1]-img_dummy_row[:, i-1])/((2*L)+1)
#    return img_dummy_col
# 
#def Guassian_MA(img, variance):
#    img2 = img.copy()
#    M, N = img.shape
#    L = 3
#    iterations = int(np.ceil(variance/4))
#    for i in range(0, iterations):
#        img_dummy = np.zeros([3*M, 3*N])
#        img_dummy = Mirror_bound(img2)         
#        Mov_Avg_output = Moving_avg_filt(img_dummy, L, M, N)
#        img2 = Mov_Avg_output.copy()
#    return img2
#
#def Reduce(img1, variance):
#     M, N = img1.shape
#     img = img1.copy()
#     img = Guassian_MA(img, variance)
#     return img[::2, ::2]
#
#def Expand(img11, img22, variance):
#    img1 = img11.copy()
#    img2 = img22.copy()
#    M1, N1 = img1.shape
#    M2, N2 = img2.shape
#    dummy = np.zeros([M1, N1])
#    dummy[::2, ::2] = img2 
#    return img1-2.8*Guassian_MA(dummy, variance)
#
#A = cv.imread('AppleGrey.jpg', 0)
#B = cv.imread('OrangeGrey.jpg', 0)
#variance = 4
## generate Gaussian pyramid for A
#G = A.copy()
#gpA = [G]
#for i in range(6):
#    G = Reduce(G, variance)
#    gpA.append(G)
#
## generate Gaussian pyramid for B
#G = B.copy()
#gpB = [G]
#for i in range(6):
#    G = Reduce(G, variance)
#    gpB.append(G)
#
## generate Laplacian Pyramid for A
#lpA = [gpA[5]]
#for i in range(5,0,-1):
#    GE = Expand(gpA[i-1], gpA[i], variance)
##    GE = cv.pyrUp(gpA[i])
##    L = cv.subtract(gpA[i-1],GE)
##    L = gpA[i-1] - GE
#    lpA.append(GE)
#
## generate Laplacian Pyramid for B
#lpB = [gpB[5]]
#for i in range(5,0,-1):
#    GE = Expand(gpB[i-1], gpB[i], variance)
##    GE = cv.pyrUp(gpB[i])
##    L = cv.subtract(gpB[i-1],GE)
##    L = gpB[i-1] - GE
#    lpB.append(GE)
#
## Now add left and right halves of images in each level
#LS = []
#for la,lb in zip(lpA,lpB):
#    rows,cols = la.shape
#    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
#    LS.append(ls)
## now reconstruct
#ls_ = LS[0]
#for i in range(1,6):
#    ls_ = cv.pyrUp(ls_)
#    ls_ = cv.add(ls_, LS[i])
## image with direct connecting each half
#real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
#cv.imshow('Pyramid_blending2.jpg',ls_)
#cv.imshow('Direct_blending.jpg',real)




import cv2 as cv
import numpy as np
A = cv.imread('AppleGrey.jpg')
B = cv.imread('OrangeGrey.jpg')
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)
# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i-1],GE)
    lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i-1],GE)
    lpB.append(L)
# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)
# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])
# image with direct connecting each half
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
cv.imshow('Pyramid_blending2.jpg',ls_)
cv.imshow('Direct_blending.jpg',real)

