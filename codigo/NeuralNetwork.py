# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:13:37 2016

@author: gbayomi
"""

import numpy as np
from scipy import optimize
from pylab import *

class NeuralNetwork(object):
    def __init__(self, inputSize, hiddenLayerUnits, outputSize, activation, regularization=0):
        #Init the input, hidden and output layers sizes
        self.inputSize = inputSize
        self.hiddenLayerUnits = hiddenLayerUnits
        self.outputSize = outputSize

        #Init the regularization factor (0 if none) and the chosen activation function string
        self.regularization = regularization
        self.activation = activation

    def loadRandomWeights(self):
        #Load random weights from 0 to 1
        self.W1 = np.random.rand((self.inputSize+1),self.hiddenLayerUnits)
        self.W2 = np.random.rand((self.hiddenLayerUnits+1),self.outputSize)

        #Resize the weights from -1 to 1
        self.W1 = 2*self.W1 - 1
        self.W2 = 2*self.W2 - 1

    def loadWeights(self, W1, W2):
        #Load arbitrary weights
        self.W1 = W1
        self.W2 = W2

    def forwardPropagation(self, X):
        #Append the bias term for the input
        X = appendOnes(X)
        #Multiply the input by the weights and get the respective activation function values
        self.z2 = np.dot(X, self.W1)
        self.a2 = activation(self.z2, self.activation)
        #Append the bias term for the hidden layer
        self.a2 = appendOnes(self.a2)
        #Multiply the hidden layer by the weights and get the respective activation function values
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = activation(self.z3, self.activation)
        return self.yHat
        
    def getCost(self, X, Y):
        #Compute the cost function for X,Y
        self.yHat = self.forwardPropagation(X)
        J = 0.5*sum((Y-self.yHat)**2) + (self.regularization/2)*(sum(self.W1**2)+sum(self.W2**2))
        return J
        
    def getGradient(self, X, Y):
        #Compute the gradient of W1 and W2 for a given X and Y:
        self.yHat = self.forwardPropagation(X)
        ###IMPORTANT: Add the bias term again
        X = appendOnes(X)

        #Delta for the hidden-output layer
        delta3 = np.multiply(-(Y-self.yHat), activationPrime(self.z3, self.activation))
        dJdW2 = np.dot(self.a2.T, delta3) + self.regularization*self.W2

        #Delta for the input-hidden layer
        delta2 = np.delete(np.dot(delta3, self.W2.T),-1,1)*activationPrime(self.z2, self.activation)
        dJdW1 = np.dot(X.T, delta2) + self.regularization*self.W1
        
        return dJdW1, dJdW2
        
    def backPropagation(self, X, Y, learning_rate):
        #Update W1 and W2 using the gradient descent
        dJdW1, dJdW2 = self.getGradient(X, Y)
        self.W1 = self.W1 - learning_rate*dJdW1
        self.W2 = self.W2 - learning_rate*dJdW2
        
    def train(self, X, Y, learning_rate, K):
        #Init the cost function array
        J = np.zeros(K)

        #Run the backprop algorithm K times
        for i in range(0, K):
            self.backPropagation(X, Y, learning_rate)
            J[i] = self.getCost(X, Y)
        print "Iterations: " + str(K) 
        print "Cost: " + str(self.getCost(X, Y))
        return J

    def getAccuracy(self, X, Y):
        #Get the percentage of correct predictions
        self.yHat = self.forwardPropagation(X)
        c = 0
        for i in range(0, len(Y)):
            if np.argmax(Y[i]) == np.argmax(self.yHat[i]):
                c = c+1
        return "Accuracy: " + str(100*c/float(len(Y))) + "%"
    
#### Static Methods
def appendOnes(z):
    #append the bias term for a 'z' array
    onesArray = np.ones((len(z),1))
    z = np.append(z, onesArray, axis=1)
    return z

def activation(z, function):
    #The activation function
    if function == "sigmoid":
        return 1.0/(1.0+np.exp(-z))
    elif function == "tanh":
        return np.tanh(z)

def activationPrime(z, function):
    #Derivative of the activation function
    if function == "sigmoid":
        return activation(z, "sigmoid")*(1-activation(z, "sigmoid"))
    elif function == "tanh":
        return 1/(np.cosh(z)**2)
    