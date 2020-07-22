# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam

from keras.regularizers import l2

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from PIL import Image
from imutils import paths

import numpy as np


imagePaths = paths.list_images('./images')

data = []
labels = []

for imagePath in imagePaths:
    image = Image.open(imagePath)
    #resize images into 32x32 pixles and normalize [0,1] by dividing by 255.0
    image = np.array(image.resize((32,32))) / 255.0
    data.append(image)
    
    label = imagePath.split('/')[-1].split('_')[0]
    labels.append(label)
    

data=np.array(data)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25) 

# define our Convolutional Neural Network architecture
model = Sequential()
model.add(Conv2D(8, (3, 3), padding="same", kernel_regularizer=l2(0.0005), input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="same", kernel_regularizer=l2(0.0005)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(0.0005)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(8))#number of classes in the dataset
model.add(Activation("softmax"))

#train the model
opt = Adam(lr=1e-3, decay=1e-3/50)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX,testY), epochs=50, batch_size=32)

#evaluate predictions
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

