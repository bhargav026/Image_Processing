import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#img1 = cv2.imread('subject01.centerlight.pgm',cv2.IMREAD_GRAYSCALE)
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
        fint = fint*(fint>0)
    return fint

dim = [64,64]
iLoc = [(34,3),(16,20),(8,4)]
img = np.zeros(dim)
for (x,y) in iLoc:
    img[x,y] = 1

th = range(0,179,2)
sup = 2*int(dim[0]/np.sqrt(2))
sono = np.zeros([sup,len(th)])


for i in range(len(th)):
    ang = th[i]*np.pi/180
    for j in range(sup):
        t1 = j-sup/2
        p=0
        for r in range(sup):
            r1 = r-sup/2
            x = r1*np.sin(ang)+t1*np.cos(ang)
            y = t1*np.sin(ang) - r1*np.cos(ang)
            p=p+bilinearInt(img,x+dim[0]/2,y+dim[1]/2)
        sono[j,i] = p

plt.figure()

plt.imshow(sono,cmap='gray')
plt.title('Gradient Energy')

plt.show()
