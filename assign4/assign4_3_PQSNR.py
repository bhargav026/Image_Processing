import cv2 as cv
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
img1=cv.imread('MrBean.jpg',cv.IMREAD_GRAYSCALE)


def quantize(image,N):
    min_val = np.min(image)
    max_val = np.max(image)
    step_size = (max_val - min_val) / (2**N - 1)
    quantized_image = np.round((image - min_val) / step_size) * step_size + min_val
    return quantized_image 

def onClick():
    m=i.get()
    print(m)
    
    if m==1:  # P=N=M
        y=quantize(img1,1)
        #y.show()
        x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
        SQNR=20*np.log10(255/x)
        print(SQNR)
    elif m==2: #P=N=2M
         y=quantize(img1,2)
         #y.show()
         x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
         SQNR=20*np.log10(255/x)
         print(SQNR)
    elif m==3: #2P=N=M
         y=quantize(img1,3)
         #y.show()
         x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
         SQNR=20*np.log10(255/x)
         print(SQNR)
    elif m==4: #P=N=M/2
         y=quantize(img1,4)
         #y.show()
         x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
         SQNR=20*np.log10(255/x)
         print(SQNR)
    elif m==5: #P=N=M/2
         y=quantize(img1,5)
         #y.show()
         x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
         SQNR=20*np.log10(255/x)
         print(SQNR)
    elif m==6: #P=N=M/2
         y=quantize(img1,6)
         #y.show()
         x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
         SQNR=20*np.log10(255/x)
         print(SQNR)
    elif m==7: #P=N=M/2
         y=quantize(img1,7)
         #y.show()
         x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
         SQNR=20*np.log10(255/x)
         print(SQNR)
    elif m==8: #P=N=M/2
         y=quantize(img1,8)
         #y.show()
         x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
         SQNR=20*np.log10(255/x)
         print(SQNR)
    elif m==9: #P=N=M/2
         y=quantize(img1,9)
         #y.show()
         x=np.sqrt(np.sum(np.square(img1-y)))/np.sqrt(362340)
         SQNR=20*np.log10(255/x)
         print(SQNR)


pqsnr = []
for i in range(1,10):
    quantized_image = quantize(img1,i)
    pqsnr.append(20*np.log10(255/(np.sqrt(np.sum(np.square(img1-quantized_image))+0.1)/np.sqrt(362340))))
    #0.1 is added to avoid Zero divios error at n=8 , 255 levels exist. so the differenr between images is 0
    # it resukts the PQSNR is inf. we can not plot. so by adding 0.1 , we get a high value



root=Tk()
i=IntVar()

r1=Radiobutton(root,text='1',value=1,variable=i,command=onClick)
r2=Radiobutton(root,text='2',value=2,variable=i,command=onClick)
r3=Radiobutton(root,text='3',value=3,variable=i,command=onClick)
r4=Radiobutton(root,text='4',value=4,variable=i,command=onClick)
r5=Radiobutton(root,text='5',value=5,variable=i,command=onClick)
r6=Radiobutton(root,text='6',value=6,variable=i,command=onClick)
r7=Radiobutton(root,text='7',value=7,variable=i,command=onClick)
r8=Radiobutton(root,text='8',value=8,variable=i,command=onClick)
r9=Radiobutton(root,text='9',value=9,variable=i,command=onClick)
r1.pack()
r2.pack()
r3.pack()
r4.pack()
r5.pack()
r6.pack()
r7.pack()
r8.pack()
r9.pack()
root.mainloop()

plt.plot([1,2,3,4,5,6,7,8,9], pqsnr)
#cv.destroyAllWindows()

    