import map_of_neurons as nr
from Loader import *
from NeuralNetwork import * 
import time

weights = np.load('files_test/weights.npy')
position = np.load('files_test/position.npy')

#Format the training images to [0, 1) size
test_images, test_labels = getTestingSampleXY(10000)
test_images = test_images/255.0

#Format the training images to [0, 1) size
train_images, train_labels = getSampleXY(60000)
train_images = train_images/255.0

#get neurons
neuronMap = nr.MapOfNeurons(15, 784, 100)
neuronMap.loadMap(weights, position)

def cosine_similarity(u,v):
	dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
	return dist


#10 posicoes que estao mais proximas e usar como input => 30
X_type_2 = np.zeros(shape=(60000, 30))
for i in range(0, len(train_images)):
	temp = np.zeros(30)
	ids = neuronMap.getNBestNeuronsID(train_images[i], 10)
	euclideans = neuronMap.euclidean(neuronMap.weightArray, train_images[i])
	for k in range(0, len(ids)):
		position = neuronMap.getPositionById(ids[k])
		temp[3*k] = position[0]
		temp[3*k + 1] = position[1]
		temp[3*k + 2] = euclideans[ids[k]]
	X_type_2[i] = temp

X_test_type_2 = np.zeros(shape=(60000, 30))
for i in range(0, len(test_images)):
	temp = np.zeros(30)
	ids = neuronMap.getNBestNeuronsID(test_images[i], 10)
	euclideans = neuronMap.euclidean(neuronMap.weightArray, test_images[i])
	for k in range(0, len(ids)):
		position = neuronMap.getPositionById(ids[k])
		temp[3*k] = position[0]
		temp[3*k + 1] = position[1]
		temp[3*k + 2] = euclideans[ids[k]]
	X_test_type_2[i] = temp


#Generate a Neural Network: input size -> 784; hidden layer->50; output->10; activation-> sigmoid; regularization-> 0.001
NN = NeuralNetwork(30, 10, 10, "sigmoid", 0.001)
#Load random weights from (-1, 1)
NN.loadRandomWeights()
#Repeat backprop algorithm 1200 times

print "start"
start_time = time.time()
J = NN.train(X_type_2, train_labels, 0.0003, 1000)
print NN.getAccuracy(X_test_type_2, test_labels)
print("--- %s minutes ---" % ((time.time() - start_time)/60.0))

#Save binary data
np.save('neural_4/fileW1.npy', NN.W1)
np.save('neural_4/fileW2.npy', NN.W2)
np.save('neural_4/fileJ.npy', J)

