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
    def __init__(self, inputNodes, nInputs, nOutputs, step, momentum):
        self.inputNodes = inputNodes
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.step = step
        self.predictions = []
        self.deltaOutputs = np.zeros([nOutputs, 1])
        self.deltaHidden = np.zeros([nOutputs, 1])
        self.delWeight = np.zeros([nOutputs, nInputs + 1])
        self.momentum = momentum
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
    # Update weights
    # Compute delta for the output layer
    def computeDeltaOutput(self):
        self.deltaOutputs =  self.activations * (1-self.activations) * (self.target - self.activations)

    # Compute delta for the hidden layer
    def computeDeltaHidden(self, outputLayer):
        # loop on hidden nodes without bias and calculate 
        # delta with respect to output layer delta
        self.deltaHidden = self.activations * (1 - self.activations) * np.dot(np.transpose(outputLayer.weightsArray), outputLayer.deltaOutputs)      

    # Compute delta weights in the outputLayer and update weights
    def updateOutputWeights(self, hiddenLayer):
        self.delWeight = self.step * np.transpose(np.dot(np.transpose([hiddenLayer.activations]), [self.deltaOutputs])) + momentum * self.delWeight
        self.weightsArray += self.delWeight

    # Compute weights in hidden layer and update weights
    def updateHiddenWeights(self):
        self.delWeight = self.step * np.dot(np.transpose([self.deltaHidden[:-1]]), [self.inputNodes]) + momentum * self.delWeight
        self.weightsArray += self.delWeight

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
    
def train(inputImg, target ,hiddenLayer, outputLayer):
    # Set the target value tk for output unit k to 0.9 if the input class is the kth class, 0.1 otherwise.
    labelNormalized = np.where(target > 0, 0.9, 0.1)
    outputLayer.setTarget(labelNormalized)

    # Set input to hidden layer
    hiddenLayer.setInput(inputImg)

    # Add bias node to hidden layer
    hiddenLayer.addBiasInput()

    # Compute hidden layer activations
    hiddenLayer.forwardProp()

    # Add Bias on activations
    hiddenLayer.addBiasActivations()

    # Compute output layer activiations
    outputLayer.setInput(hiddenLayer.activations)

    outputLayer.forwardProp()

    # Compute deltaK
    outputLayer.computeDeltaOutput()

    # Compute deltaJ
    hiddenLayer.computeDeltaHidden(outputLayer)

    # Update Output weights
    outputLayer.updateOutputWeights(hiddenLayer)

    # Update Hidden weights
    hiddenLayer.updateHiddenWeights()

def getAccuracy(trainImgs, trainLabels, hiddenLayer, outputLayer, confusionActual, confusionPrediction):
    Trues = 0
    confusionPrediction = []
    confusionActual = []
    for img in range(trainImgs.shape[0]):

        # Set input to hidden layer
        hiddenLayer.setInput(trainImgs[img])

        # Add bias node to hidden layer
        hiddenLayer.addBiasInput()

        # Compute hidden layer activations
        hiddenLayer.forwardProp()

        # Add Bias on activations
        hiddenLayer.addBiasActivations()

        # Compute output layer activiations
        outputLayer.setInput(hiddenLayer.activations)

        outputLayer.forwardProp()

        # Softmax
        outputLayer.softmax()

        # Calculate Accuracy
        prediction = np.argmax(outputLayer.predictions)

        actual = np.argmax(trainLabels[img]).astype(int)

        if(prediction == actual):
            Trues += 1
        confusionPrediction.append(prediction)

        confusionActual.append(actual)

    return Trues / trainImgs.shape[0]



# Importing training dataset
dataset_train = pd.read_csv('train.csv')
labels_train = dataset_train.iloc[0:15000, 0].values
picture_train = dataset_train.iloc[0:15000,1:785].values

# Importing test dataset
dataset_test = pd.read_csv('test.csv')
labels_test = dataset_test.iloc[0:1000, 0].values
picture_test = dataset_test.iloc[0:1000,1:785].values

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
n = 20
step = 0.1
momentum = 0.9

# Initialize Layers
hiddenLayer = Layer(picture_train[0], 784, n, step, momentum)
outputLayer = Layer(picture_train[0], n , 10, step, momentum)

# Arrays storing accuracies
trainAccuracyArray = []
testAccuracyArray = []
confusionPrediction = []
confusionActual = []

epochs = 10
for epoch in range(0, epochs):
    trainTrues = 0
    for img in range(picture_train.shape[0]):
        train(picture_train[img], label_train[img], hiddenLayer, outputLayer)

    # first epoch training done on all train data  
    # get train and test Accuracy  
    trainAccuracy = getAccuracy(picture_train, label_train,hiddenLayer, outputLayer, confusionActual, confusionPrediction)    
    testAccuracy = getAccuracy(picture_test, labels_test, hiddenLayer, outputLayer, confusionActual, confusionPrediction)
    print("Epoch #" + str(epoch+1) + ": train accuracy: " + str(trainAccuracy)+ ": test accuracy: " + str(testAccuracy))
print("End")