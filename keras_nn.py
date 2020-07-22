# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np

from imutils import paths
from PIL import Image

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
    
 
data=np.array(data)

#encode labels as 1-hot vectors
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25) 


#define the 6-3-3-8 architecture using Keras
#input layer: 6 nodes (6 features)
#2 hidden layers: 3 nodes each
#output layer: 8 nodes (8 classes)

model = Sequential()
model.add(Dense(3, input_shape=(6,), activation="sigmoid"))
model.add(Dense(3, activation="sigmoid"))
model.add(Dense(8, activation="softmax"))

#train teh model using SGD
opt = SGD(lr=0.1, momentum=0.9, decay = 0.1/250)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=250, batch_size=16)

#evaluate the network
predictions = model.predict(testX, batch_size=16)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))