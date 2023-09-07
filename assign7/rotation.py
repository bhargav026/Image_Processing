import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



def bilinearInt(img,x,y):
    dim = img.shape
    if(int(x)<0 or int(x)>=(dim[0]-1) or int(y)<0 or int(y)>=(dim[1]-1)):
        fint = 0
    else:
        dfx = int(img[int(x)+1,int(y)])-int(img[int(x),int(y)])
        dfy = int(img[int(x),int(y)+1])-int(img[int(x),int(y)])
        dx = x-int(x)
        dy = y-int(y)
        fint = int(img[int(x),int(y)])+dfx*dx+dfy*dy
    return fint

def rotImg(img,th):
    th = np.pi*th/180
    dim = img.shape
    rImg = np.zeros([3*dim[0],3*dim[1]])
    rMat = [[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]]
    for i in range(3*dim[0]):
        for j in range(3*dim[1]):
            x = i-3*dim[0]//2
            y = j-3*dim[1]//2
            r = rMat@np.array([x,y])+[dim[0]//2,dim[1]//2]
            rImg[i,j] = bilinearInt(img,r[0],r[1])
    return rImg


img = cv2.imread('AppleGrey.jpg',cv2.IMREAD_GRAYSCALE)
rImg = rotImg(img,10)


plt.figure()
plt.imshow(rImg,cmap='gray')
plt.title('Gradient Energy')

plt.show()



