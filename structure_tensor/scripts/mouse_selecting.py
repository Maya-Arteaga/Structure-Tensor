#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:04:28 2023

@author: juanpablomayaarteaga
"""

import cv2
import numpy as np
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from skimage.segmentation import clear_border


#Path
i_path="/Users/juanpablomayaarteaga/Desktop/Garza_Lab/Morfina/"
o_path="/Users/juanpablomayaarteaga/Desktop/Output_Images/Pruebas/"



# Create a list to store the points of the ROI polygon
roi_points = []

# Define a mouse callback function to capture mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the clicked point to the list of ROI points
        roi_points.append((x,y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Clear the list of ROI points
        roi_points.clear()

# Load the image
image= cv2.imread("/Users/juanpablomayaarteaga/Desktop/Hand_crop.tif")
#image=cv2.imread("/Users/juanpablomayaarteaga/Desktop/Garza_Lab/Morfina/Iba.tif")

#Iba= cv2.imread("/Users/juanpablomayaarteaga/Desktop/Hand_crop.tif")
#image=Iba[:,:,2]

# Create a window to display the image
cv2.namedWindow('Image')

# Set the mouse callback function for the window
cv2.setMouseCallback('Image', mouse_callback)

# Loop until the user presses 'q'
while True:
    # Make a copy of the input image
    image_copy = image.copy()
    
    # If there are at least three ROI points, draw the polygon on the image
    if len(roi_points) >= 3:
        roi_points_array = np.array(roi_points)
        cv2.fillPoly(image_copy, [roi_points_array], (0,255,0))
    
    # Display the image
    cv2.imshow('Image', image_copy)
    
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    
    # If the 'q' key is pressed, exit the loop
    if key == ord('q'):
        break

# If there are at least three ROI points, create a mask and apply it to the input image
if len(roi_points) >= 3:
    mask = np.zeros_like(image[:,:,0])
    roi_points_array = np.array([roi_points])
    cv2.fillPoly(mask, roi_points_array, 255)
    roi = cv2.bitwise_and(image, image, mask=mask)

    # Display the ROI
    cv2.imshow('ROI', roi)
    cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()


###############1)Area

# Threshold the image to obtain a binary image
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
ret, binary_image = cv2.threshold(roi, 1, 255, cv2.THRESH_BINARY)
plt.imshow(binary_image)

# Find the contours of the binary image
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the area of the contours
total_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    total_area += area

# Print the area
print('Area:', total_area)



#################2)Counting


#Preprocessing

roi=roi[:,:,2]

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


gammaImg = gammaCorrection(roi, 0.5)
#plt.imshow(gammaImg)
plt.imsave(o_path+"roi_gamma.jpg", gammaImg)


kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(gammaImg,cv2.MORPH_OPEN,kernel, iterations = 1)
#plt.imshow(opening)


#cv2.imshow('Original image', roi)
#cv2.imshow('Gamma corrected image', gammaImg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()







#Setting a threshold
ret, thresh = cv2.threshold(opening, 28, 255, cv2.THRESH_BINARY)
plt.imshow(thresh)
plt.imsave(o_path+"thresh_gamma.jpg", thresh)    




labels, num_cells = label(thresh)
fig, ax = plt.subplots()
ax.imshow(thresh, cmap="gray")

for i in range(1, num_cells+1):
    y, x = np.where(labels == i)
    xc = (x.max() + x.min())/2
    yc = (y.max() + y.min())/2
    ax.text(xc, yc, str(i), color="r", fontsize=8, ha="center", va="center")
    ax.set_title(f"{num_cells} cells")
plt.savefig(o_path+"Cell_counter.png")   
print(f'Total number of cells: {num_cells}')
