from random import randint
import numpy as np
import math
import sys
import numpy as np
from scipy import optimize
from pylab import *
import time
import map_of_neurons as nr
from Loader import *

#Format the training images to [0, 1) size
X, Y = getSampleXY(60000)
X = X/255.0

weights = np.load('files_test/weights.npy')
position = np.load('files_test/position.npy')

neuronMap = nr.MapOfNeurons(15, 784, 100)
neuronMap.loadMap(weights, position)
groups = neuronMap.getGroups(X, Y)

#print groups[0]
#print most_common(groups[0])


def most_common(lst):
	if not lst:
		s = '-'
		return s
	else:
		r = max(set(lst), key=lst.count)
		return str(r)

for i in range(0, 15):
	string = ''
	for j in range(0, 15):
		string = string + most_common(groups[(i*15) + j]) + '    '
	print string + '\n'
