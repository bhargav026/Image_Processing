# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:26:45 2019

@author: Madhu
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path='C:\\Users\\Personal\\.spyder-py3\\yale-face-database'
image_paths = [os.path.join(path, f) for f in os.listdir(path)]
images = []

for image_path in image_paths:
    image_pil = Image.open(image_path).convert('L')
    image = np.array(image_pil, 'uint8')
    images.append(image)
    
A=np.matrix.flatten(images[0])   
for image in images[1:]:
    col_dummy = np.matrix.flatten(image)
    A=np.c_[A,col_dummy]
    
Mean=A.sum(axis=1)/len(images)
A_mean=np.transpose(np.transpose(A)-Mean)

u, s, vh = np.linalg.svd(A_mean, full_matrices=False)
u = u[:, :20]
y = np.transpose(u) @ A_mean[:, 0]
z = u @ y
z = np.reshape(z, (243, 320))

plt.figure()
plt.subplot(121)
plt.imshow(images[0], cmap= 'gray')

plt.subplot(122)
plt.imshow(z, cmap = 'gray')
plt.show()