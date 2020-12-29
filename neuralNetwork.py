"""
    ECE 4424 - Project Classify Image Using 2-layers Neural Network with MNIST data set
    Minh Nguyen
    11/20/2020
    File implement the 3 layers neural network for MNIST Classification
    About:  File implement stochastic gradient descent learning algorithm to feedForward neuralentwork.
            Gradients are calculated using backpropagation instead of the derivative of error rate in lecture 33
"""
import random
import numpy 
import time
########################################################### Helper methods: Calculate Sigmoid
def sigmoid(input):
    return 1.0/(1.0 + numpy.exp(-input))

########################################################### Helper method: Calculate Derivative of Sigmoid
def sigmoidDer(input):
    return (numpy.exp(-input)) /( (1.0+numpy.exp(-input)) * (1.0+numpy.exp(-input)) )

"""
    Class Network: Create 2 layers that can specify number of nodes/neurons in the hidden layers
                   Input layer is default to be 784 neurons (28x28)
                   Output layer is default to be 10 neurons (0-9)
"""
class Network(object):
    ########################################################### Init
    """
        Function init():
        @param:     self: The neural network itself, like "this" in C++ or Java
                    sizes: Number of neurons in each layers in the netowk
                    biases: initialize randomly. We won't set biases for those neurons since the biases are only ever used in computing the outputs from the later layers
                    weights: initialize randomly
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(i, 1) for i in sizes[1:]] # initialize the bias for the middle and last layer only
        
        # n = all elements in array but the last one; m = all elements in the array but the first one
        self.weights = [numpy.random.randn(m, n) for n, m in zip(sizes[:-1], sizes[1:])] 

    ########################################################## Cost Function For Output Layer
    """
        Function cost function between the calculated labels array with the actual labels in the training set
        @param:     output_activation:  a 10x1 array contain activation between 0-1 of the last layer
                    y: a 10x1 array contain the actual labels 0 or 1.
    """
    def costOutputNeurons(self, outputActivations, y):
        return(outputActivations - y)

    ########################################################### Back Propagation
    """
        Function backpropagation
        About: Return a tuple (gradBias, gradWeight) representing the gradient for the cost function 
        @param: self:   The neural network itself
                input_activation:      input activation
                labels:      activation array of the actual output layer - given
    """
    def backPropagation(self, input_activation , labels):

        # gradBias and gradWeight are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights
        gradBias = [numpy.zeros(b.shape) for b in self.biases]
        gradWeight = [numpy.zeros(w.shape) for w in self.weights]

        # feedForward to calculate the activation of all neuron in the network
        activation = input_activation 
        activations = [input_activation] # store all activation

        sigmoidInputArr = [] # list to store all the sigmoidInputArr vectors, layer by layer. sigmoidInput = activation * weight + biases
        for b, w in zip(self.biases, self.weights):
            sigmoidInput = numpy.dot(w, activation) + b
            sigmoidInputArr.append(sigmoidInput)
            activation = sigmoid(sigmoidInput) 
            activations.append(activation) # update activation of the previous layer
                    
        # updating the activation, weight, and biase
        delta = self.costOutputNeurons(activations[-1], labels) * sigmoidDer(sigmoidInputArr[-1])  # error in output layer
        gradBias[-1] = delta 
        gradWeight[-1] = numpy.dot(delta, activations[-2].transpose())

        # layer = 1 means the last layer of neurons. layer = 2 is the second-last layer
        for layer in range(2, self.num_layers):
            sigmoidInput = sigmoidInputArr[-layer]
            delta = numpy.dot(self.weights[-layer + 1].transpose(), delta) * sigmoidDer(sigmoidInput)
            gradBias[-layer] = delta
            gradWeight[-layer] = numpy.dot(delta, activations[-layer - 1].transpose())

        return (gradBias, gradWeight)

    ########################################################### Update Mini Batch
    """
        Function updateMiniBatch
        About:  update the network's weights and biases by applying gradient descent using backpropagation to single mini batch.
        @param: self = the network object
                miniBatch = for stochastic gradient descent 
                eta = the learning rate 
    """
    def updateMiniBatch(self, miniBatch, eta):
        gradBias = [numpy.zeros(b.shape) for b in self.biases] # initialize gradient of bias as zeros
        gradWeight = [numpy.zeros(w.shape) for w in self.weights] # initialize gradient of weight as zeros
        
        for input, label in miniBatch:
            delta_gradBias, delta_gradWeight = self.backPropagation(input,label) # Calculate back propagation delta calulated_labels an actual labesl in each mini batch
            gradBias = [currGradBias + deltaBias for currGradBias, deltaBias in zip(gradBias, delta_gradBias)] # Update the biases array
            gradWeight = [currGradWeight + deltaWeight for currGradWeight, deltaWeight in zip(gradWeight, delta_gradWeight)] # Update the weight array
        
        # Provide the final weights and biases, zip allow to interate element bby element though tuple
        self.weights = [w-(eta/len(miniBatch))*gw for w, gw in zip(self.weights, gradWeight)]
        self.biases = [b-(eta/len(miniBatch))*gb for b, gb in zip(self.biases, gradBias)]

    ########################################################### Stochastic Gradient Descent
    """
        Function StochasticGD: train the neural network using mini-batch stochastic gradient descent.
        This is supervised learning since the training features and labels are given. We only test on the test dataset
        @param:     self: the neural network itself
                    training_data: list of tuples (x,y) representing the training inputs and the desired outputs
                    testing_data: is provided then the network will evaluated against the test data after each epoch, and the partial progress printed out.
                    eta = learning rate
                    We predetermine the miniBatch_size say 100 images
    """
    def StochasticGD(self, training_data, testing_data, epochs, miniBatchSize, eta):
        start_time_total = time.time()

        training_data = list(training_data)
        number_of_train = len(training_data)
        testing_data = list(testing_data)
        number_of_test = len(testing_data)
        
        for i in range(epochs): 
            #numpy.random.shuffle(training_data)
            mini_batches = [training_data[k:k+miniBatchSize] for k in range(0,number_of_train,miniBatchSize)] # Separate big batch in to mini batches
            mini_batches = numpy.array(mini_batches)
            for batch in mini_batches:
                self.updateMiniBatch(batch, eta) # update the weights and biases
            print(f"========== Prediction {i} ==========")
            print(f"Current training accuracy is: {(self.precisionCal(testing_data)/number_of_test)*100}%")
            print(f"Current training error is: {(1 - (self.precisionCal(testing_data)/number_of_test))*100}% \n")

        print(f"The training time is: {time.time() - start_time_total} s")
            
    ########################################################### Feed Forward
    """
        Function feedForward: Return the output of the network if a is an input
        @param      acti = the activation/ the sigmoid function of the current layer
        @return     acti = activation of the output
    """
    def feedForward(self, acti):
        for bias, weight in zip(self.biases, self.weights):
            acti = sigmoid(numpy.dot(weight, acti) + bias)
        return acti

    ########################################################### Evaluation with test set
    """
        Function precisionCal
        About:  return the number of test inputs for which the neural network outputs the correct results.
                Note that the neural networks's output is assumed to be the index of whichever neuron in the final layer has the highest activation
        @param:     self: the neural network itself
                    testing_data: Testing features and labels
    """
    def precisionCal(self, testing_data):
        cal_results = [(numpy.argmax(self.feedForward(acti_input)), label) for (acti_input, label) in testing_data]
        # accuracy = 0 
        # for (cal_labels,labels) in cal_results:
        #     if(cal_labels == labels):
        #         accuracy += 1
        # return accuracy
        test_results = [(numpy.argmax(self.feedForward(x)), y) for (x, y) in testing_data]
        return sum(int(x == y) for (x, y) in test_results)
