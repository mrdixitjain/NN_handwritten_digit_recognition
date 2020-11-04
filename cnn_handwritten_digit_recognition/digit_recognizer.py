import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn import preprocessing
import os
from os import listdir
from matplotlib import image
from sklearn.model_selection import train_test_split
import cv2
import time

from PIL import Image


def pause() :
	a = input("press enter to continue")

data = pd.read_csv("train.csv")

output = data['label']
data.drop('label', inplace=True, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(data, output, test_size=0.20, random_state=42)
print(len(X_train), len(X_test))

# lb = preprocessing.LabelBinarizer()
# lb.fit(output)

# y_test = lb.transform(y_test)
# y_train = lb.transform(y_train)


X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_test = np.array(y_test)
# y_test = y_test.T
y_train = np.array(y_train)
# y_train = y_train.T

print()
print(y_test.shape)
print(y_train.shape)
print(X_test.shape)
print(X_train.shape)

X_train = X_train/255
X_test = X_test/255

from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(.5))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))


# parameters, costs = nn_model(X_train, y_train, n_h = 25, learning_rate = 1.2, max_itr = 10000, print_cost=True)
# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn import preprocessing
import os
from os import listdir
from matplotlib import image
from sklearn.model_selection import train_test_split
import cv2
import time

from PIL import Image


def pause() :
	a = input("press enter to continue")

data = pd.read_csv("train.csv")

output = data['label']
data.drop('label', inplace=True, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(data, output, test_size=0.20, random_state=42)

# lb = preprocessing.LabelBinarizer()
# lb.fit(output)

# y_test = lb.transform(y_test)
# y_train = lb.transform(y_train)


X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_test = np.array(y_test)
# y_test = y_test.T
y_train = np.array(y_train)
# y_train = y_train.T

print()
print(y_test.shape)
print(y_train.shape)
print(X_test.shape)
print(X_train.shape)

X_train = X_train/255
X_test = X_test/255

from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(.5))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))


# parameters, costs = nn_model(X_train, y_train, n_h = 25, learning_rate = 1.2, max_itr = 10000, print_cost=True)
# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

del X_test
del X_train
del y_train
del y_test


test = pd.read_csv("test.csv")

test = np.array(test)

test = test.reshape(test.shape[0], 28, 28, 1)/255

for i in range(5) :
	# print("Predicted : " + str(predictions[i]))
	img = test[i]*255
	img = np.reshape(img, (28, 28))

	img1 = np.reshape(img, (1, 28, 28, 1))
	predictions = model.predict(img1)
	predict = np.argmax(predictions, axis=1)
	fig = plt.figure()
	plt.imshow(img)
	fig.suptitle("Predicted : " + str(predict), fontsize=10)
	plt.show()


rahul
harshal
abhinav
rashi
keval
yash

deepak