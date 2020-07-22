# -*- coding: utf-8 -*-
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import median, threshold_minimum, gaussian, laplace, frangi, hessian, meijering, sato
from skimage.morphology import disk

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import random


def get_features(matrix,pixles,features,index):
    for z in range(len(pixles)):
        x,y = pixles[z]
        features[z][index] = matrix[x][y]

    return features

#%% read image

image = data.immunohistochemistry()
plt.imshow(image)
image = rgb2gray(image)
plt.imshow(image, cmap="gray")

#noise removal (median filter is an edge preserver (performs better than Gaussian))
med = median(image, disk(5))
plt.imshow(med, cmap="gray")

#threasholding
thresh_min = threshold_minimum(med)
binary_min = med > thresh_min

#invert image (make the cells white and background black)
inverted = np.invert(binary_min)

plt.imshow(inverted, cmap="gray")

#%% sample positive pixles (512 pixles)

pos_pixles = set()
random.seed(0)

while len(pos_pixles) < 1000:
    x = random.randint(0,511)
    y = random.randint(0,511)
    
    if inverted[x][y] == True:
        pos_pixles.add((x,y))

neg_pixles = set()

while len(neg_pixles) < 1000:
    x = random.randint(0,511)
    y = random.randint(0,511)
    
    if inverted[x][y] == False:
        neg_pixles.add((x,y))

pixles = list(pos_pixles) + list(neg_pixles)
Y = [1 for x in range(len(pos_pixles))] + [0 for x in range(len(neg_pixles))]

#%% generate the features

features = [[0 for i in range(7)] for j in range(len(pixles))]

features = get_features(gaussian(image),pixles,features,0)
features = get_features(laplace(image),pixles,features,1)
features = get_features(median(image),pixles,features,2)
features = get_features(frangi(image),pixles,features,3)
features = get_features(hessian(image),pixles,features,4)
features = get_features(meijering(image),pixles,features,5)
features = get_features(sato(image),pixles,features,6)

#%% split the data into training and testing

X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.30, stratify=Y, random_state=0)

#%% train the models
model = RandomForestClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)

#%% model performance
acc = accuracy_score(y_test,pred)
print(acc)

#%% generate features for the entire image
pixles = [(x,y) for x in range(512) for y in range(512)]

features = [[0 for i in range(7)] for j in range(len(pixles))]

features = get_features(gaussian(image),pixles,features,0)
features = get_features(laplace(image),pixles,features,1)
features = get_features(median(image),pixles,features,2)
features = get_features(frangi(image),pixles,features,3)
features = get_features(hessian(image),pixles,features,4)
features = get_features(meijering(image),pixles,features,5)
features = get_features(sato(image),pixles,features,6)

#%% apply the model to the entire image
pred_image = [[0 for i in range(512)] for j in range(512)]

pred = model.predict(features)

#%% generate the new image

count = 0

for count in range(len(pixles)):
    x,y = pixles[count]
    pred_image[x][y] = pred[count]
    
#%% show predicted image
plt.imshow(pred_image, cmap="gray")

