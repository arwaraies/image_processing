# -*- coding: utf-8 -*-

import numpy as np

from imutils import paths
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report


def extract_color_stats(image):
    (R, G, B) = image.split() #split image into three color channels
    
    #each image is represented with three features: mean and standard diviation of each color channel
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)] 
    
    return features


imagePaths = paths.list_images('./images')
data = []
labels = []

for imagePath in imagePaths:
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)
    
    label = imagePath.split('/')[-1].split('_')[0]
    labels.append(label)
    

le = LabelEncoder()
labels = le.fit_transform(labels) #converts labels from categories to numbers

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25) 

models = [KNeighborsClassifier(n_neighbors=1), GaussianNB(), LogisticRegression(solver="lbfgs", multi_class="auto"), SVC(kernel="linear"), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100), MLPClassifier()]

for model in models:
    print(model)
    model.fit(trainX,trainY)
    predictions = model.predict(testX)
    print(classification_report(testY, predictions, target_names=le.classes_))
    
