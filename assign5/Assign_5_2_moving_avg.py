# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:22:29 2019

@author: Personal
"""

#%matplotlib notebook
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons,Slider

# %matplotlib notebook
#import numpy as np
#import cv2 as cv
#from matplotlib import pyplot as plt
#import matplotlib.image as mpimg
#from matplotlib.widgets import RadioButtons,Slider

def padding(n,t):
    out=np.zeros((3*n.shape[0],3*n.shape[1]))
    if(t==1):
        out[n.shape[0]:2*n.shape[0],n.shape[1]:2*n.shape[1]]=n
    
    elif(t==2):
        n1=np.fliplr(n)
        m=np.concatenate((n1,n,n1), axis=1)
        m1=np.flipud(m)
        out=np.concatenate((m1,m,m1), axis=0)
        
    elif(t==3):
        for i in range(0,3*n.shape[1],n.shape[1]):
            for j in range(0,3*n.shape[0],n.shape[0]):
                out[j:j+n.shape[0],i:i+n.shape[1]]=n
    return out  

def movingavg(m,L):
    
    nn1=m[input_image.shape[0]-L:2*input_image.shape[0]+L,input_image.shape[1]-L:2*input_image.shape[1]+L]
    
    nn2=np.zeros((input_image.shape[0],input_image.shape[1]+2*L))
    
    nn3=np.zeros((input_image.shape[0],input_image.shape[1]))
    
    nn2[0:1,:]=np.sum(nn1[0:2*L+1,:],axis=0)
    for i in range(L+1,input_image.shape[0]+L+1,1):
        nn2[i:i+1,:]=nn2[i-1:i,:]+nn1[i+L:i+L+1,:]-nn1[i-L-1:i-L,:]
        
    nn3[:,0]=np.sum(nn2[:,0:2*L+1],axis=1)
    
    for i in range(L+1,input_image.shape[1]+L+1,1):
        nn3[:,i:i+1]=nn3[:,i-1:i]+nn2[:,i+L:i+L+1]-nn2[:,i-L-1:i-L]
        
    nn3 = nn3/(L+1)**2
    nn3 = (nn3-np.min(nn3))/(np.max(nn3)-np.min(nn3))
    return nn3*255
    
input_image= cv.imread('BaboonGrayscale.jpg',cv.IMREAD_GRAYSCALE)
t=3
a=padding(input_image,t)
plt.figure()
plt.subplot(121)
input,_,_ = plt.imshow(a, cmap='gray', vmin=0, vmax=255), plt.xticks([]),plt.yticks([])
l=10
output_image=movingavg(a,l)
plt.subplot(122)
out,_,_ = plt.imshow(output_image, cmap='gray'),plt.xticks([]),plt.yticks([])



filterrad=RadioButtons(plt.axes([0.01,0.2,0.15,0.2]),('1.zero','2.mirror','3.periodic'),active=1)
ax1 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
length = Slider(ax1, 'length', 3,19, valinit=5,valstep=2)

def go(label):
    
    global t
    global input_image
    global a
    global l
    t= int(label[0])
    a=padding(input_image,t)
    c=movingavg(a,int(l/2))
    input.set_data(a)
    out.set_data(c)
    plt.draw()

def go1(label):
    global l
    global a
    l=int(label)
    c= movingavg(a,int(l/2))
    out.set_data(c)
    plt.draw()
    
filterrad.on_clicked(go)
length.on_changed(go1)