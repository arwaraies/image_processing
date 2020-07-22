# -*- coding: utf-8 -*-
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.filters import median, threshold_li, threshold_local, try_all_threshold, threshold_li
from skimage.morphology import disk, black_tophat, white_tophat, binary_opening, binary_closing
from skimage.filters.rank import tophat
from skimage import measure

from sklearn.cluster import KMeans, DBSCAN, FeatureAgglomeration, SpectralClustering

import matplotlib.pyplot as plt

import numpy as np

#%% read image

image = imread("./counting/data/raw/001cell.png")
image = rgb2gray(image)
plt.imshow(image, cmap="gray")

#%% noise removal (median filter is an edge preserver (performs better than Gaussian))

med = median(image, disk(5))
plt.imshow(med, cmap="gray")

#%% remove background

background = white_tophat(med,disk(5))
plt.imshow(background, cmap="gray")

#%% can threshold only gray images
fig, ax = try_all_threshold(background, verbose=False)
plt.show()

#%% threasholding
li_thresh = threshold_li(background)
binary_li = background > li_thresh

#invert image (make the cells white and background black): bad for labelling
#inverted = np.invert(binary_li)
plt.imshow(binary_li, cmap="gray")

#%% post-process thresholding

opening = binary_opening(binary_li)

plt.imshow(opening, cmap="gray")

#%% label opjects

label_image = measure.label(opening)
plt.imshow(label_image)
table = measure.regionprops_table(label_image, properties = ['area','eccentricity','minor_axis_length','major_axis_length','label'])
plt.savefig('labelled_image.png')

#%% generate the feature vectors of each object

X = []
objects = max(table['label'])

for x in range(objects):
    X.append([table['area'][x],table['eccentricity'][x],table['minor_axis_length'][x],table['major_axis_length'][x]])

#%% cluster the objects

clustering = KMeans(n_clusters=4, random_state=0).fit(X)
#clustering = DBSCAN(eps=20, min_samples=2).fit(X)
#clustering = SpectralClustering(n_clusters=2).fit(X)
labels = clustering.labels_

#%% generate the predicted image

pred_image = [[0 for i in range(len(label_image[0]))] for j in range(len(label_image))]

for i in range(len(pred_image)):
    for j in range(len(pred_image[0])):
        if label_image[i][j] != 0:
            if labels[label_image[i][j]-1] == 2:
                #pred_image[i][j] = labels[label_image[i][j]-1]+1
                pred_image[i][j] = 1
            else:
                pred_image[i][j] = 2

plt.figure(2)        
plt.imshow(pred_image, cmap="gray")
plt.savefig('predicted_cells.png')

