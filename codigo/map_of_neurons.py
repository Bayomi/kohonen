import math
import sys
import numpy as np
from scipy import optimize
from pylab import *
import time


class MapOfNeurons(object):

	def __init__(self, size, vector_length, iteration_reference):
		self.size = size
		size_square = math.pow(size, 2)
		self.weightArray = np.random.rand(size_square, vector_length)
		self.positionArray = np.zeros(shape=(size_square, 2))
		self.setMap()
		self.sigma = 1
		self.learning_rate = 0.65
		self.iteration_count = 0
		self.iteration_reference = iteration_reference + 1

	def loadMap(self, weights, position):
		self.weightArray = weights 
		self.positionArray = position

	def getGroups(self, inputs, labels):
		groups = [[] for x in xrange(0,225)]
		for i in range(0, len(inputs)):
			input = inputs[i]
			id = self.getBestNeuronID(input)
			groups[id].append(np.argmax(labels[i]))
		return groups

	def setMap(self):
		for i in range(0, self.size):
			for j in range(0, self.size):
				self.positionArray[(i*self.size) + j][0] = i
				self.positionArray[(i*self.size) + j][1] = j

	def getPositionById(self, id):
		return self.positionArray[id]

	def getNeighborsArray(self, id):
		euclidean_distances = self.getDistancesFromNeuron(id)
		gaussian_values = self.gaussian(euclidean_distances)
		return gaussian_values

	def getDistancesFromNeuron(self, neuron_reference_id):
		ref_position = self.getPositionById(neuron_reference_id)
		euclidean_distances = self.euclidean(self.positionArray, ref_position)
		return euclidean_distances

	def gaussian(self, distances):
		gaussian = np.exp(-1*np.power((distances/self.sigma), 2)/2)
		return gaussian

	def euclidean(self, a, b):
		euclidean = np.sqrt(np.sum(np.square(a - b),axis=1))
		return euclidean

	def updateSigma(self):
		self.sigma = self.sigma*0.975

	def updateLearning_rate(self):
		count = self.iteration_count
		reference = self.iteration_reference
		percentage = count/float(reference)
		self.learning_rate = self.learning_rate*math.exp(-percentage*1.5)

	def iterate(self):
		self.iteration_count = self.iteration_count + 1
		self.updateSigma()
		self.updateLearning_rate()

	def getNBestNeuronsID(self, input, n):
		euclidean_distance = self.euclidean(self.weightArray, input)
		id = euclidean_distance.argsort()[:n]
		
		return id

	def getBestNeuronID(self, input):
		euclidean_distance = self.euclidean(self.weightArray, input)
		id = argmin(euclidean_distance)
		return id

	def updateWeights(self, id, input):
		neighborhood_diff = np.array([self.getNeighborsArray(id)]).T
		sub_diff = input - self.weightArray 
		diff = sub_diff*neighborhood_diff
		self.weightArray = self.weightArray + self.learning_rate*diff
