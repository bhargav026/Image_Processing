import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import cv2 as cv

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.35)
ax1=plt.axes([0.25, 0.15, 0.5, 0.02], facecolor='lightgoldenrodyellow')
Variance_of_guassian = Slider(ax1,'Variance_of_guassian_fil',  0.1, 30, valinit=0.1, valstep=0.5)
ax2=plt.axes([0.25, 0.05, 0.5, 0.02], facecolor='lightgoldenrodyellow')
Levels = Slider(ax2,'Levels', 1, 4, valinit=1, valstep=1)
img1=cv.imread('VanGoghGray.tif',0)
plt.subplot(121)
plt.imshow(img1, cmap='gray', vmin =0, vmax =255 )
plt.show()
plt.title('Input image')

def Mirror_bound(img):
    M,N=img.shape
    img_dummy=np.zeros([3*M,3*N])
    img_copy=img.copy()
    for k in range(0,3):
        for l in range(0,3):
            img_temp=np.zeros([M,N])
            if (k==0 and not(l==1)) or (k==2 and not(l==1)):
                img_temp=np.flip(np.flip(img_copy,1),0)
            elif (k==1 and not(l==1)):
                img_temp=np.flip(img_copy,1)
            elif (l==1 and not(k==1)):
                img_temp=np.flip(img_copy,0)
            else:
                img_temp=img_copy
            img_dummy[k*M:(k*M)+M:,l*N:(l*N)+N:]=img_temp 
            
    return img_dummy

def Moving_avg_filt(img_dummy1,L,M,N):
    img_dummy=img_dummy1.copy()
    img_dummy_row=np.zeros([M,N+2*L])
    img_dummy_col=np.zeros([M,N])
    img_dummy1=img_dummy[M-L:2*M+L,N-L:2*N+L]

    img_mask=img_dummy1[0:(2*L)+1,:]
    img_dummy_row[0,:]=np.sum(img_mask,axis=0)/((2*L)+1)
    
    for i in range(1,M-1):
        img_dummy_row[i,:]=img_dummy_row[i-1,:]+(img_dummy1[i+2*L+1,:]-img_dummy1[i-1,:])/((2*L)+1)
   
    img_mask1=img_dummy_row[:,0:(2*L)+1]
    img_dummy_col[:,0]=np.sum(img_mask1,axis=1)/((2*L)+1)
    for i in range(1,N-1):
        img_dummy_col[:,i]=img_dummy_col[:,i-1]+(img_dummy_row[:,i+2*L+1]-img_dummy_row[:,i-1])/((2*L)+1)
    return img_dummy_col

def Guassian_MA(img,variance):
    img2=img.copy()
    M,N=img.shape
    L=3
    iterations=int(np.ceil(variance/4))
    for i in range(0,iterations):
        img_dummy=np.zeros([3*M,3*N])
        img_dummy=Mirror_bound(img2)         
        Mov_Avg_output=Moving_avg_filt(img_dummy,L,M,N)
        img2=Mov_Avg_output.copy()
    
    return img2

def Reduce(img1,variance):
     M,N=img1.shape
     img=img1.copy()
     img=Guassian_MA(img,variance)
     return img[::2,::2]
     
def Guassian_pyramid(img1,variance,levels):
    dummy1=img1.copy()
    M,N=img1.shape
    Guassian_list=[]
    Guassian_list.append(img1)
    for i in range(1,int(levels+1)):
        dummy=Reduce(dummy1,variance)
        Guassian_list.append(dummy)
        dummy1=dummy

    return Guassian_list

def Expand(img11,img22,variance):
    img1=img11.copy()
    img2=img22.copy()
    M1,N1=img1.shape
    M2,N2=img2.shape
    upsampled_image=np.zeros([M1,N1])
    upsampled_image[::2,::2]=img2
    return img1-3.8*Guassian_MA(upsampled_image,variance)

def Laplacian_pyramid(Guassian_pyramid_list,variance):
    Laplacian_pyrd=[]
    for i in range(0,len(Guassian_pyramid_list)-1):
        Detail_img=Expand(Guassian_pyramid_list[i],Guassian_pyramid_list[i+1],variance)
        Laplacian_pyrd.append(Detail_img)
    Laplacian_pyrd.append(Guassian_pyramid_list[-1])
    
    return Laplacian_pyrd

def Laplacian_pyrd_recons(Laplacian_pyramid_list,variance):
    dummy=Laplacian_pyramid_list[len(Laplacian_pyramid_list)-1]
    for i in range(0,len(Laplacian_pyramid_list)-1):
        dummy=Expand(Laplacian_pyramid_list[len(Laplacian_pyramid_list)-i-2],-1*dummy,variance)
    
    return dummy

def update(val):
    variance=Variance_of_guassian.val
    levels=Levels.val
    Guassian_pyramid_list=Guassian_pyramid(img1,variance,levels)
    Laplacian_pyramid_list=Laplacian_pyramid(Guassian_pyramid_list,variance)
    for i in range(0,int(levels+1)):
        plt.subplot(2,5,i+1)
        plt.imshow(np.float64(Guassian_pyramid_list[i]), cmap='gray')
        plt.show()
        plt.title('level ' + str(i))
        plt.subplot(2,5,i+6)
        plt.imshow(np.float64(Laplacian_pyramid_list[i]), cmap='gray')
        plt.show()
        plt.title('level ' + str(i))

    reconstructed_img=Laplacian_pyrd_recons(Laplacian_pyramid_list,variance)
    plt.figure()
    plt.imshow(np.float64(reconstructed_img), cmap='gray')
    plt.show()

Variance_of_guassian.on_changed(update)
Levels.on_changed(update)
