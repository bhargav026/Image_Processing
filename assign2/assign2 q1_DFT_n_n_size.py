import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
img1= cv2.imread('cameraman.tif',cv2.IMREAD_GRAYSCALE)
img2= cv2.imread('zelda.png',cv2.IMREAD_GRAYSCALE)
radio = RadioButtons(plt.subplot(241), ('64', '128', '256','512','1024'), active=0)

def dft(m):
    f1 = np.fft.fft2(img1,s=([m,m]))
    f2 = np.fft.fft2(img2,s=([m,m]))
    fshift1 = np.fft.fftshift(f1)
    magnitude_spectrum1 =20*np.log(np.abs(fshift1))
    phase_spectrum1 = np.angle(fshift1)
    
    fshift2 = np.fft.fftshift(f2)
    magnitude_spectrum2 = 20*np.log(np.abs(fshift2))
    phase_spectrum2 = np.angle(fshift2)
    return(magnitude_spectrum1,phase_spectrum1,magnitude_spectrum2,phase_spectrum2)

def update(label):
    
    m = int(label,10)
    magnitude_spectrum1,phase_spectrum1,magnitude_spectrum2,phase_spectrum2=dft(m)
    mag1.set_data(magnitude_spectrum1)
    ph1.set_data(phase_spectrum1)
    mag2.set_data(magnitude_spectrum2)
    ph2.set_data(phase_spectrum2)
    plt.draw()

magnitude_spectrum1,phase_spectrum1,magnitude_spectrum2,phase_spectrum2=dft(64)

plt.subplot(242),plt.imshow(img1, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(243)
mag1=plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(244)
ph1=plt.imshow(phase_spectrum1, cmap='gray')
plt.title('phase spectrum'),plt.xticks([]), plt.yticks([])

plt.subplot(246),plt.imshow(img2, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.subplot(247)
mag2=plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.subplot(248)
ph2=plt.imshow(phase_spectrum2, cmap='gray')
plt.xticks([]), plt.yticks([])
    
radio.on_clicked(update)

plt.show()
