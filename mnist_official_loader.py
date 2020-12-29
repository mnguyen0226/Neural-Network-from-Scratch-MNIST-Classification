"""
    ECE 4424 - Project Classify Image Using 2-layers Neural Network with MNIST data set
    Minh Nguyen
    12/7/2020
    About:  module importing local database 
            module preprocessing data by reshaping and turn from [0-255] pixel range into [0-1] pixel range
    The database can be found in http://yann.lecun.com/exdb/mnist/
    About Database: 
        60000 training 28x28 pixels image with 60000 respecting labels for testing purposes
        10000 testing 28x28 pixels image with 10000 respecting labels for training purposes
"""
import numpy

########################################################### Main Data Preproccessing Methods
"""
    Function Processing Data by calling extracting methods above then zip the training feature and label in a tuple and testing features and label in another tuple
    @return     training_data:  zip tuple of training features and training labels
                testing_data:   zip tuple of testing features and testing labesl
"""
def processData():
    training_features = extract_feature('mnistDB/train-images.idx3-ubyte', 60000) 
    training_labels = extract_labels_training('mnistDB/train-labels.idx1-ubyte')

    testing_features = extract_feature('mnistDB/t10k-images.idx3-ubyte', 10000)
    testing_labels = extract_labels_testing('mnistDB/t10k-labels.idx1-ubyte')

    # zip the processed data. zip return an iterable object, but making a list will allow to call each object instead of iterate thru it
    training_data = list(zip(training_features, training_labels))
    testing_data = list(zip(testing_features, testing_labels))

    return(training_data, testing_data)

########################################################### Helper methods
"""
    Import binary data and export respective matrix
    @param:  idxByteFile         input database including the magic number with bytes corresponding to each number
             numSize             number of numbers that you want to withdraws from the DB for training or testing
    @return: imageTensor         3D array with numSizex28x28 - stacks of 28x28 images 
"""
def extract_feature(idxByteFile, numSize):
    with open(idxByteFile, 'rb') as binaryFile:
        binaryList = binaryFile.read() # import list of binary values
    binaryList = binaryList[16:] # delete magic number
    decimalList = [] # Create list for decimals 0-255 value

    # Loop through each "number" in the list of binary value
    for number in binaryList:
        decimalList.append(int(number))
    
    pixels = numpy.array(decimalList) # Converts decimal list to numpy array for arithmetics
    pixels = pixels / 256.0 # Resize 0-255 to 0-1 ranges for ez backprop
    imageTensor = pixels.reshape((numSize, 784, 1)) # Converts 1D into 3D tensor of images

    return imageTensor # return tensor of images

"""
    Function return a 10x1 array with 1.0 in the ith position and zeros everywhere else. 
    This function is used for convert single number into binary array for training the backprop in neural network
"""
def vectorized_arr(i):
    zeroArr = numpy.zeros((10, 1)) #initilize vector zero
    zeroArr[i] = 1.0 # number in position ith will be 1
    return zeroArr 

"""
    Import training labels. Since I did back propagation and the output layers contains for 10 neurons/node
        To be able to train labels, we have to set the labels into arrays of 10 of binary with 1 appear in the position of the values
    @param:  idxByteFile         input database including the magic number with bytes corresponding to each number
    @return: vectorized_labels:  array of 10x1 arrays
"""
def extract_labels_training(idxByteFile):
    """
        Input: idx file containing labels value
        Output: labels = list of image labels
    """
    with open(idxByteFile, 'rb') as binaryFile:
        binaryList = binaryFile.read() # import list of binary values
    binaryList = binaryList[8:]  # delete magic number
    decimalList = [] # Create list for decimals 0-255 value

    # Loop through each "number" in the list of binary value
    for number in binaryList:
        decimalList.append(int(number))

    labels = numpy.array(decimalList) # convert to array for arithmetic
    vectorized_labels = [vectorized_arr(i) for i in labels] 
    
    return vectorized_labels

"""
    For testing labels, we don't need to vectorize it, the vectorize process will be converted in the validation process
    @param:  idxByteFile         input database including the magic number with bytes corresponding to each number
    @param:  labels:             array of labels for checking purposes
"""
def extract_labels_testing(idxBinaryFile):
    with open(idxBinaryFile, 'rb') as binaryFile:
        binaryList = binaryFile.read() # Get list of binary values
    binaryList = binaryList[8:] # Delete the magic number
    decimalList = []

    for number in binaryList:
        decimalList.append(int(number))

    labels = numpy.array(decimalList)

    return labels

