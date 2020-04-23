# Importing the libraries
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from scipy.special import softmax

class Layer:
    # Constructor
    def __init__(self, inputNodes, nInputs, nOutputs, step):
        self.inputNodes = inputNodes
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.predictions = []

        # insert random weights including bias
        self.weightsArray = np.random.uniform(low=-0.05, high=0.05, size=(nOutputs, nInputs + 1))

    # Forward Propagation method
    def forwardProp(self):
        self.activations = np.dot(self.inputNodes ,np.transpose(self.weightsArray))
        self.sigmoid()
    
    # Compute Sigmoid (Squash output nodes)
    def sigmoid(self):
        self.activations = 1 / (1 + np.exp(-self.activations))

    # Compute Softmax to get PDF
    def softmax(self):
        self.predictions = softmax(self.activations)

    # Back Propagation method
    # Compute lambda for the layer
    # Update weights
    # Compute delta weights

    # Input nodes setter
    def setInput(self, inputNodes):
        self.inputNodes = inputNodes

    # Add bias to input nodes
    def addBiasInput(self):
        self.inputNodes = np.append(self.inputNodes, -1)

    # Add bias to activation nodes
    def addBiasActivations(self):
        self.activations = np.append(self.activations, -1)




# Importing training dataset
dataset_train = pd.read_csv('train.csv')
labels_train = dataset_train.iloc[0:15000, 0].values
picture_train = dataset_train.iloc[0:15000,1:785].values

# Importing test dataset
dataset_test = pd.read_csv('test.csv')
labels_test = dataset_test.iloc[:, 0].values
picture_test = dataset_test.iloc[:,1:785].values

# Scale inputs
picture_train = picture_train/255
picture_test = picture_test/255

# encode labels
ohe = OneHotEncoder()
labels_train = labels_train.reshape(-1, 1)
ohe.fit(labels_train)
label_train = ohe.transform(labels_train)
label_train = label_train.toarray()

labels_test = labels_test.reshape(-1, 1)
ohe.fit(labels_test)
labels_test = ohe.transform(labels_test)
labels_test = labels_test.toarray()

# Number of hidden nodes
n = 3

# Initialize Layers
hiddenLayer = Layer(picture_train[0], 784, n, 0.01)
outputLayer = Layer(picture_train[0], n , 10, 0.01)

# Set input to hidden layer
hiddenLayer.setInput(picture_train[0])

# Add bias node to hidden layer
hiddenLayer.addBiasInput()

# Compute hidden layer activations
hiddenLayer.forwardProp()

# Add Bias on activations
hiddenLayer.addBiasActivations()

# Compute output layer activiations
outputLayer.setInput(hiddenLayer.activations)
outputLayer.forwardProp()

# Compute softmax
outputLayer.softmax()
print("End")