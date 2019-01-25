from keras.datasets import mnist
from keras import models
from keras.layers import *
from PIL import Image
import numpy as np
from keras.optimizers import SGD
import  tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()
trainX = []
trainY = []

for i in range(len(x_train)):
    trainX.append(x_train[i].flatten())

for i in range(len(y_train)):
    a = np.array([0 for i in range(10)])
    a[y_train[i]-1] = 1
    trainY.append(a)

trainY = np.array(trainY)
trainX = np.array(trainX)
print(trainY[-1], y_train[-1])



model = models.Sequential()
model.add(Dense(units=40, use_bias=True, input_dim=784, activation="sigmoid"))
model.add(Dense(units=10, activation="softmax"))

opt = SGD()
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

model.fit(trainX, trainY, epochs=1)



