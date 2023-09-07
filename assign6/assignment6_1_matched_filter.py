import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider
import cv2 as cv
from PIL import Image
from scipy import signal
from numpy import linalg as LA

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.35)


threshold=0.7
img=cv.imread('Nanoparticles.jpg',0)
#img=img1[0:25,0:25]
template=cv.imread('NanoTemplate.jpg',0)
plt.subplot(121)
plt.imshow(img, cmap='gray', vmin =0, vmax =255 )
plt.show()
plt.title('Input image')
plt.xticks([])
plt.yticks([])
M,N=img.shape
M1,N1=template.shape
template_norm=template/LA.norm(template)
dummy=np.zeros([M-M1+1,N-N1+1])
dummy1=np.zeros([M-M1+1,N-N1+1])
mask=np.zeros([M1,N1])

for i in range(0,M-M1+1):
    for j in range(0,N-N1+1):
        mask=img[i:i+M1,j:j+N1]
        mask_norm=mask/LA.norm(mask)
        dummy[i,j]=np.sum(mask_norm*template_norm)
       

dummy1=255*(dummy>threshold)
dummy2=np.pad(dummy1,8,'constant')
for i in range(8,398):
    for j in range(8,449):
        if(dummy2[i,j]==255):
            dummy2[i:i+8,j-8:j+8]=0
            dummy2[i,j]=255
            
    
plt.subplot(122)
plt.imshow(dummy2, cmap='gray')
plt.show()
plt.title('Output image')
plt.xticks([])
plt.yticks([])
print('No.of particles ')
print(np.sum(dummy2/255))


#for i range (1,5):
#    for j in range(1,5):
        
#x=np.array([[1,2,3],[1,2,3],[1,2,3]])
#y=np.array([[1,1,1],[1,1,1],[1,1,1]])
#z=np.array([[10,10,10],[10,10,10],[10,10,10]])
#corr = signal.correlate2d(x,y)
#print(corr)
#corr1 = signal.correlate2d(x,z)
#print(corr1)
#
#
#x1=(x)/LA.norm(x)
#y1=(y)/LA.norm(y)
#z1=(z)/LA.norm(z)
#corr2 = signal.correlate2d(x1,y1)
#print(corr2)
#corr3 = signal.correlate2d(x1,z1)
#print(corr3)
#print(np.corrcoef(x,y))
#print(np.corrcoef(x,z))

#print(corr)