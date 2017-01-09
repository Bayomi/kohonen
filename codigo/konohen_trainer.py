import math
import sys
import numpy as np
from scipy import optimize
from pylab import *
import time
import map_of_neurons as nr
from loader_mnist import *


class KonohenMap(object):

	def __init__(self, size_of_map, iterations, X):
			self.iterations = iterations
			self.inputs = X
			self.neuronMap = nr.MapOfNeurons(size_of_map, self.inputs.shape[1], iterations)

	def train(self):
		start_time = time.time()
		print "start"
		for k in range(0, self.iterations):
			for i in range(0, len(self.inputs)):
				input = self.inputs[i]
				id = self.neuronMap.getBestNeuronID(input)
				self.neuronMap.updateWeights(id, input)
			self.neuronMap.iterate()
		print("--- %s minutes ---" % ((time.time() - start_time)/60.0))


#train network
iterations = 100

X, Y = getSampleTrain(60000)
X = X/255.0
konohen = KonohenMap(15, iterations, X)
konohen.train()

#save position and weights
#Save binary data
np.save('files/weights.npy', konohen.neuronMap.weightArray)
np.save('files/position.npy', konohen.neuronMap.positionArray)


