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
from sklearn import preprocessing
from parameters import *

data = pd.read_csv("train.csv")
output = data['label']
data.drop('label', inplace=True, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(data, output, test_size=0.20, random_state=42)
print(len(X_train), len(X_test))

lb = preprocessing.LabelBinarizer()
lb.fit(output)

y_test = lb.transform(y_test)
y_train = lb.transform(y_train)


X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], -1).T

# y_test = np.array(y_test)
y_test = y_test.T
# y_train = np.array(y_train)
y_train = y_train.T

X_train = X_train/255
X_test = X_test/255

def sigmoid(z):

    s = 1 / (1 + np.exp(-z))
    
    return s

def random_initialization(n_x, n_h, n_y) :
	w1 = np.random.randn(n_h, n_x)*0.01
	b1 = np.zeros((n_h, 1))
	w2 = np.random.randn(n_y, n_h)*0.01
	b2 = np.zeros((n_y, 1))

	# print(w1.shape)
	# print(b1.shape)
	# print(w2.shape)
	# print(b2.shape)

	parameters = {"w1" : w1,
				"w2" : w2,
				"b1" : b1,
				"b2" : b2
				}

	return parameters

def propagate(parameters, X) :

	w1 = parameters['w1']
	b1 = parameters['b1']
	w2 = parameters['w2']
	b2 = parameters['b2']

	z1 = np.dot(w1, X) + b1
	a2 = np.tanh(z1)
	z2 = np.dot(w2, a2) + b2 
	a3 = sigmoid(z2)

	cache = {"z1" : z1,
			"z2" : z2,
			"a2" : a2,
			"a3" : a3
			}
	return a3, cache

def compute_cost(A2, Y) :

	m = Y.shape[1]

	log_probs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2), 1-Y)
	cost = -np.sum(log_probs)/m
	return cost

def backward_propagation(X, Y, cache, parameters) :

	w1 = parameters['w1']
	b1 = parameters['b1']
	w2 = parameters['w2']
	b2 = parameters['b2']

	z1 = cache["z1"]
	z2 = cache["z2"]
	a1 = cache['a2']
	a2 = cache['a3']

	m = Y.shape[1]

	dz2 = a2 - Y
	dw2 = np.dot(dz2, a1.T)/m 
	db2 = np.sum(dz2, axis = 1, keepdims=True)/m 

	dz1 = np.multiply(np.dot(w2.T, dz2), 1-(np.tanh(z1))**2)
	dw1 = np.dot(dz1, X.T)/m 
	db1 = np.sum(dz1, axis=1, keepdims=True)/m 

	grads = {"dw1" : dw1,
			"db1" :db1,
			"dw2" : dw2,
			"db2" : db2
			}

	return grads


def update_parameters(parameters, grads, learning_rate) :	

	w1 = parameters['w1']
	b1 = parameters['b1']
	w2 = parameters['w2']
	b2 = parameters['b2']

	dw1 = grads['dw1']
	dw2 = grads['dw2']
	db1 = grads['db1']
	db2 = grads['db2']

	w1 -= learning_rate*dw1
	w2 -= learning_rate*dw2
	b1 -= learning_rate*db1
	b2 -= learning_rate*db2

	parameters = {"w1" : w1,
				"w2" : w2,
				"b1" : b1,
				"b2" : b2
				}

	return parameters

def predict(parameters, X, Y) :

	w1 = parameters['w1']
	b1 = parameters['b1']
	w2 = parameters['w2']
	b2 = parameters['b2']

	z1 = np.dot(w1, X) + b1
	a2 = np.tanh(z1)
	z2 = np.dot(w2, a2) + b2 
	a3 = sigmoid(z2)
	predictions = np.argmax(a3, axis=0)
	y = np.argmax(Y, axis=0)
	print('\nAccuracy: ', end = str(np.mean(np.double(predictions == y)) * 100))

	x = X.T
	for i in range(5) :
		# print("Predicted : " + str(predictions[i]))
		img = x[i]*255
		img = np.reshape(img, (28, 28))
		fig = plt.figure()
		plt.imshow(img)
		fig.suptitle("Label : " + str(y[i]) + ". Predicted : " + str(predictions[i]), fontsize=10)
		plt.show()

def nn_model(X, Y, n_h, learning_rate, max_itr, print_cost) :
	n_x = X.shape[0]
	n_y = Y.shape[0]

	# print(X.shape)
	# print(Y.shape)

	parameters = random_initialization(n_x, n_h, n_y)

	costs = []

	for i in range(max_itr) :
		a3, cache = propagate(parameters, X)
		cost = compute_cost(a3, Y)
		grads = backward_propagation(X, Y, cache, parameters)
		parameters = update_parameters(parameters, grads, learning_rate)

		if i % 100 == 0:
			costs.append(cost)
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))

	print("Train accuracy : ", end = "")
	predict(parameters, X_train, y_train)

	return parameters, costs


# Build a model with a n_h-dimensional hidden layer
# parameters = nn_model(X_train, y_train, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))


# parameters, costs = nn_model(X_train, y_train, n_h = 25, learning_rate = 1.2, max_itr = 10000, print_cost=True)

parameters = parameters()

predict(parameters, X_train, y_train)
predict(parameters, X_test, y_test)