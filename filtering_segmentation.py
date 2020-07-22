# -*- coding: utf-8 -*-

#%% modules imports
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import data, util, color, measure
from skimage.filters import try_all_threshold, threshold_minimum, threshold_local, median, gaussian, sobel, roberts, scharr, prewitt
from skimage.filters.rank import tophat
from skimage.morphology import disk, black_tophat, white_tophat, binary_erosion, binary_dilation, binary_opening, binary_closing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from scipy import ndimage as ndi

import matplotlib.pyplot as plt

import numpy as np
#%% example of a local image

image = imread("./images/coast_arnat59.jpg")
plt.imshow(image)
plt.imshow(rgb2gray(image), cmap="gray")

#%% example of loading image from the data module

image = data.immunohistochemistry()
plt.imshow(image)
image = rgb2gray(image)
plt.imshow(image, cmap="gray")
#plt.savefig('original.png')

#%% thresholding example (global thresholds): thresholding is used to segment images

#can threshold only gray images
fig, ax = try_all_threshold(image, verbose=False)
plt.show()

#%% apply minimum thresholding algorithm (performs better than other global thresholds algorithms)

thresh_min = threshold_minimum(image)
binary_min = image > thresh_min

plt.figure(0)
plt.imshow(binary_min, cmap="gray")

#invert image(make the cells white and background black)
inverted = np.invert(binary_min)
plt.figure(1)
plt.imshow(inverted, cmap="gray")

#%% apply local thresholding (better for fine-grain segmentation)

block_size = 21 #must be an odd number
local_thresh = threshold_local(image, block_size)
binary_local = image > local_thresh
plt.imshow(binary_local, cmap="gray")

#%% apply a noise removal step using Median filter (non-linear filter)
#(noise removal blurres the images and improves thresholding)

#median filter is an edge preserver (performs better than Gaussian)
med = median(image, disk(5))
plt.imshow(med, cmap="gray")

#apply thresholding after noise removal
fig, ax = try_all_threshold(med, verbose=False)
plt.show()

#%% apply a noise removal step using Gaussian filter (liner Filter)

gau = gaussian(image)
plt.imshow(gau, cmap="gray")

#apply thresholding after noise removal
fig, ax = try_all_threshold(gau, verbose=False)
plt.show()

#%% apply edge detection

# apply edge detection
edges = sobel(image)
plt.figure(1)
plt.imshow(edges, cmap="gray")

# apply noise removal (using median filter) then edge detection
# can find edges of small objects and too close together
med = median(image, disk(5)) 
edges = sobel(med)
plt.figure(2)
plt.imshow(edges, cmap="gray")

# apply thresholding (using minimum filter) then edge detection
# less preferred approach
thresh_min = threshold_minimum(image)
binary_min = image > thresh_min
edges = sobel(binary_min)
plt.figure(3)
plt.imshow(edges, cmap="gray")

# apply noise removal (using median filter) then thresholding (using minimum filter) then edge detection
# can find edges of large and far away objects
med = median(image, disk(5))
thresh_min = threshold_minimum(med)
binary_min = med > thresh_min
edges = sobel(binary_min)
plt.figure(4)
plt.imshow(edges, cmap="gray")

#%% background removal (useful when there are different intensities in the background)

# apply background removal using tophat filter
background = tophat(image,disk(5))
plt.figure(1)
plt.imshow(background, cmap="gray")

# apply noise removal (using median filter) then background substraction (using tophat filter)
med = median(image, disk(5)) 
background = tophat(med,disk(5))
plt.figure(2)
plt.imshow(background, cmap="gray")

#%% post processing thresholding using binary closing (fill holes) and binary openning

# apply noise removal
med = median(image, disk(5)) 

# apply thresholding
thresh_min = threshold_minimum(med)
binary_min = med > thresh_min

inverted = np.invert(binary_min)

plt.figure(0)
plt.imshow(inverted, cmap="gray")

dialation = binary_erosion(inverted)
#dialation = binary_erosion(dialation)
#dialation = binary_erosion(dialation)
#dialation = binary_erosion(dialation)
#dialation = binary_erosion(dialation)
#dialation = binary_erosion(dialation)
#dialation = binary_erosion(dialation)
#dialation = binary_erosion(dialation)
#dialation = binary_erosion(dialation)

plt.figure(1)
plt.imshow(dialation, cmap="gray")

dialation = binary_dilation(dialation)
dialation = binary_dilation(dialation)
dialation = binary_dilation(dialation)
dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)
#dialation = binary_dilation(dialation)

plt.figure(2)
plt.imshow(dialation, cmap="gray")

# add watershed algorithm
# calculate distance map
distance = ndi.distance_transform_edt(dialation)

# calculate markers at the maximuma of the distance map
local_maxi = peak_local_max(distance, labels=dialation, footprint=np.ones((3, 3)), indices=False)
markers = ndi.label(local_maxi)[0]
#
## apply watershed 
#labels = watershed(-distance, markers, mask=inverted)  

labels = watershed(-distance, markers, mask=dialation)  
plt.figure(3)
plt.imshow(labels)

#%% watershed algorithm for segmentation

# apply noise removal
med = median(image, disk(5)) 

# apply thresholding
thresh_min = threshold_minimum(med)
binary_min = med > thresh_min

inverted = np.invert(binary_min)

# calculate distance map
distance = ndi.distance_transform_edt(inverted)

# calculate markers at the maximuma of the distance map
local_maxi = peak_local_max(distance, labels=inverted, footprint=np.ones((3, 3)), indices=False)
markers = ndi.label(local_maxi)[0]

# apply watershed 
labels = watershed(-distance, markers, mask=inverted)  

plt.figure(0)
plt.imshow(labels, cmap="Blues")

#apply edge detection
edges = sobel(labels)
plt.figure(2)
plt.imshow(edges, cmap="gray")
plt.savefig('four_objects.png')

#%% waterershed algorithm (wronge approach, but worse)

# apply noise removal
med = median(image, disk(5)) 

# apply thresholding
thresh_min = threshold_minimum(med)
binary_min = med > thresh_min

inverted = np.invert(binary_min)

# find edges
edges = sobel(inverted)

grid = util.regular_grid(inverted.shape, n_points=468)
seeds = np.zeros(inverted.shape, dtype=int)
seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1

labels = watershed(edges, seeds, compactness=True)
#plt.imshow(labels, cmap="gray")
plt.imshow(color.label2rgb(labels, inverted, bg_label=-1))


#%% image segmentation
# typical pipeline for image segmentation
# 1. preprocessing the image by noise removal
# 2. apply background removal (if necessary: i.e., there is large variation in image intensity)
# 3. apply thresholding
# 4. post processing thresholding (if necessary) binary closing (fill holes) adn binary openning
# 5. watershed to cut connected objects
# 5. post processing watershed (if necessary) binary closing (fill holes) adn binary openning


#%% connected components analysis (can also be done with watershed)
#noise removal
med = median(image, disk(5))
#thresholding
thresh_min = threshold_minimum(med)
binary_min = med > thresh_min
#invert white and black
#inverted = np.invert(binary_min)
#connected componenet labelling
label_image = measure.label(binary_min)
plt.imshow(label_image)

table = measure.regionprops_table(label_image, properties = ['area','centroid','label'])
