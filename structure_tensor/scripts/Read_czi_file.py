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
plt.imsave(o_path+"Iba.png", Iba)




import os
import skimage
from skimage import morphology, measure
import numpy as np
import csv

# Set directory path for microglia images
microglia_dir = o_path

# Set CSV filename and path
csv_filename = 'microglia_properties.csv'
csv_path = os.path.join(microglia_dir, csv_filename)

# Write headers to CSV file
headers = ['Image', 'Microglia', 'Number of branches', 'Average branch length', 'Volume', 'Surface area', 'Solidity']
with open(csv_path, mode='w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(headers)

    # Get list of microglia image files
    microglia_files = sorted([os.path.join(microglia_dir, f) for f in os.listdir(microglia_dir) if f.endswith('.tif')])

    # Loop over each image in the stack
    for i in range(len(microglia_files)):
        # Load microglia image
        microglia_image = skimage.io.imread(microglia_files[i])

        # Preprocess microglia image
        microglia_image = skimage.filters.gaussian(microglia_image, sigma=1)
        microglia_thresh = skimage.filters.threshold_otsu(microglia_image)
        microglia_binary = microglia_image > microglia_thresh

        # Remove small objects
        microglia_clean = morphology.remove_small_objects(microglia_binary, min_size=50)

        # Calculate branching features for each microglia cell
        microglia_label = measure.label(microglia_clean)
        microglia_properties = measure.regionprops(microglia_label)
        for p in microglia_properties:
            microglia_cell = microglia_label == p.label
            microglia_skeleton = morphology.skeletonize(microglia_cell)
            microglia_skeleton = morphology.thin(microglia_skeleton)
            branching_points = morphology.branch_points(microglia_skeleton)
            end_points = morphology.endpoints(microglia_skeleton)

            # Calculate the number of branches for each cell
            branches = sum([tuple(b) in p.coords for b in branching_points])

            # Calculate the length of each branch for each cell
            branch_lengths = []
            for bp in branching_points:
                distances = []
                for ep in end_points:
                    distances.append(np.linalg.norm(bp - ep))
                branch_lengths.append(max(distances))
            if branch_lengths:
                mean_branch_length = np.mean(branch_lengths)
            else:
                mean_branch_length = 0

            # Calculate other properties of microglia cell
            volume = p.area  # volume is equivalent to the area in 2D
            surface_area = p.perimeter
            solidity = p.solidity

            # Write properties of microglia cell to CSV file
            row = [i, p.label, branches, mean_branch_length, volume, surface_area, solidity]
            writer.writerow(row)



