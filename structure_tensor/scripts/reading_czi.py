#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:55:51 2023

@author: juanpablomayaarteaga
"""

import czifile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

#Path
i_path="/Users/juanpablomayaarteaga/Desktop/"
o_path="/Users/juanpablomayaarteaga/Desktop/Output_Images/Pruebas/Stack/"


"""
#Reading the image
img= cv2.imread(i_path+"Morfina.tif")
Iba= img[:,:,2]
#plt.imshow(Iba)
"""

# open the CZI file
with czifile.CziFile(i_path+"Morfina.czi") as czi:
    # get the pixel data as a numpy array
    data = czi.asarray()

    # display one of the image planes using Matplotlib
    plt.imshow(data[0, 0, 0, 0, 1, :, :], cmap='gray')
    plt.show()
    
    
# loop through the z-planes and save each image as a TIF file
for t in range(data.shape[3]):
    for c in range(data.shape[2]):
        for i in range(data.shape[1]):
            for z in range(data.shape[4]):
                # display the image using Matplotlib
                plt.imshow(data[0, c, i, t, z, :, :], cmap='gray')
                plt.axis('off')
                # save the image as a PNG file
                plt.savefig(o_path+f'image_t{t}_c{c}_i{i}_z{z}.tif', bbox_inches='tight', pad_inches=0)
                plt.clf() # clear the figure for the next image

img= cv2.imread(o_path+"image_t0_c0_i0_z13.tif")
Iba= img[:,:,2]
Neun=img[:,:,1]
Sepa= img[:,:,0]
plt.imshow(img)
plt.imshow(Neun)
plt.imshow(Iba)
plt.imshow(Sepa)



"""    
    
# loop through the z-planes and display each image using Matplotlib
for t in range(data.shape[3]):
    for c in range(data.shape[2]):
        for i in range(data.shape[1]):
            for z in range(data.shape[4]):
                # display the image using Matplotlib
                plt.imshow(data[0, c, i, t, z, :, :], cmap='gray')
                plt.show()



""""

# display all stacks as individual images using matplotlib
for z in range(num_stacks):
    plt.imshow(pixel_data[z,:,:], cmap='gray')
    plt.show()

# display all stacks as a montage using matplotlib
montage = np.transpose(pixel_data, (1,2,0)) # change the array order to (height, width, stack)
plt.imshow(montage, cmap='gray')
plt.show()