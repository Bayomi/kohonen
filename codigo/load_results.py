from NeuralNetwork import * 
from Loader import * 
import matplotlib.pyplot as plt
import map_of_neurons as nr
import time

#original entries
#Format the training images to [0, 1) size
test_images, test_labels = getTestingSampleXY(10000)
test_images = test_images/255.0

#Format the training images to [0, 1) size
train_images, train_labels = getSampleXY(60000)
train_images = train_images/255.0

#weights and positions for konohen
weights = np.load('files_test/weights.npy')
position = np.load('files_test/position.npy')

#get neurons of Konohen
neuronMap = nr.MapOfNeurons(15, 784, 100)
neuronMap.loadMap(weights, position)


#4.1
X_test_type_1 = np.zeros(shape=(10000, 784))
for i in range(0, len(test_images)):
	id = neuronMap.getBestNeuronID(test_images[i])
	X_test_type_1[i] = neuronMap.weightArray[id]

W1 = np.load('neural_1/fileW1.npy')
W2 = np.load('neural_1/fileW2.npy')
J = np.load('neural_1/fileJ.npy')

#Load Neural Net
NN = NeuralNetwork(784, 30, 10, "sigmoid")
NN.loadWeights(W1, W2)
label = NN.getAccuracy(X_test_type_1, test_labels)

plt.plot(J)
plt.xlabel(label)
plt.show()

"""
#4.2
X_type_2 = np.zeros(shape=(60000, 225))
for i in range(0, len(train_images)):
	distances = neuronMap.euclidean(neuronMap.weightArray, train_images[i])
	X_type_2[i] = distances

X_type_2 = X_type_2/amax(X_type_2)

X_test_type_2 = np.zeros(shape=(10000, 225))
for i in range(0, len(test_images)):
	distances = neuronMap.euclidean(neuronMap.weightArray, test_images[i])
	X_test_type_2[i] = distances

X_test_type_2 = X_test_type_2/amax(X_type_2)

W1 = np.load('neural_2/fileW1.npy')
W2 = np.load('neural_2/fileW2.npy')
J = np.load('neural_2/fileJ.npy')

#Load Neural Net
NN = NeuralNetwork(225, 10, 10, "sigmoid")
NN.loadWeights(W1, W2)
label = NN.getAccuracy(X_test_type_2, test_labels)

plt.plot(J)
plt.xlabel(label)
plt.show()
"""

"""

#4.3
X_test_type_3 = np.zeros(shape=(60000, 2))
for i in range(0, len(test_images)):
	temp = np.zeros(2)
	ids = neuronMap.getNBestNeuronsID(test_images[i], 1)
	for k in range(0, len(ids)):
		position = neuronMap.getPositionById(ids[k])
		temp[2*k] = position[0]
		temp[2*k + 1] = position[1]
	X_test_type_3[i] = temp 

W1 = np.load('neural_3/fileW1.npy')
W2 = np.load('neural_3/fileW2.npy')
J = np.load('neural_3/fileJ.npy')

#Load Neural Net
NN = NeuralNetwork(225, 10, 10, "sigmoid")
NN.loadWeights(W1, W2)
label = NN.getAccuracy(X_test_type_3, test_labels)


plt.plot(J)
plt.xlabel(label)
plt.show()

"""

"""
#4.4
X_test_type_4 = np.zeros(shape=(60000, 30))
for i in range(0, len(test_images)):
	temp = np.zeros(30)
	ids = neuronMap.getNBestNeuronsID(test_images[i], 10)
	euclideans = neuronMap.euclidean(neuronMap.weightArray, test_images[i])
	for k in range(0, len(ids)):
		position = neuronMap.getPositionById(ids[k])
		temp[3*k] = position[0]
		temp[3*k + 1] = position[1]
		temp[3*k + 2] = euclideans[ids[k]]
	X_test_type_4[i] = temp 

W1 = np.load('neural_4/fileW1.npy')
W2 = np.load('neural_4/fileW2.npy')
J = np.load('neural_4/fileJ.npy')

#Load Neural Net
NN = NeuralNetwork(30, 10, 10, "sigmoid")
NN.loadWeights(W1, W2)
label = NN.getAccuracy(X_test_type_4, test_labels)


plt.plot(J)
plt.xlabel(label)
plt.show()
"""

