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
        self.deltaOutputs = np.zeros([nOutputs, 1])
        self.deltaHidden = np.zeros([nOutputs, 1])
        # insert random weights including bias
        self.weightsArray = np.random.uniform(low=0.1, high=0.1, size=(nOutputs, nInputs + 1))

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
    # Update weights
    # Compute delta for the output layer
    def computeDeltaOutput(self):
        for k in range(self.activations.size):
            self.deltaOutputs[k] =  self.activations[k] * (1-self.activations[k]) * (self.target[k] - self.activations[k])

    def computeDeltaHidden(self, outputLayer):
        # loop on hidden nodes without bias and calculate 
        # delta
        for j in range(self.activations.size - 1):
            sumK = 0
            for k in range(outputLayer.activations.size):
                sumK += outputLayer.weightsArray[k][j] * outputLayer.deltaOutputs[k]
            self.deltaHidden[j] = self.activations[j] * (1 - self.activations[j]) * sumK     

    # Input nodes setter
    def setInput(self, inputNodes):
        self.inputNodes = inputNodes
    
    def setTarget(self, target):
        self.target = target

    # Add bias to input nodes
    def addBiasInput(self):
        self.inputNodes = np.append(self.inputNodes, 1)

    # Add bias to activation nodes
    def addBiasActivations(self):
        self.activations = np.append(self.activations, 1)




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
n = 2
picture_train = [[1,0]]
label_train = [[0.9]]
# Initialize Layers
hiddenLayer = Layer(picture_train[0], 2, n, 0.01)
outputLayer = Layer(picture_train[0], n , 1, 0.01)
outputLayer.setTarget(label_train[0])

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

#Compute deltaK
outputLayer.computeDeltaOutput()

#Compute deltaJ
hiddenLayer.computeDeltaHidden(outputLayer)
# Compute softmax
outputLayer.softmax()
print("End")