"""
    Program that load trained model net2
    Command: python ./testNet2.py on terminal
"""
import pickle
import numpy
import matplotlib.pyplot as plt 
import matplotlib as mpl


import mnist_official_loader 
from mnist_official_loader import processData

def main():

    training_data, testing_data = mnist_official_loader.processData() # Data Preprocessing

    print("========Second Run: [784,100,10] 30 epochs=========\n")    
    pickleFile = "trained_models/trainedModel_784_100_10_30Epoch.pkl"
    
    # Load Trained Data 
    with open(pickleFile, 'rb') as file:
        net = pickle.load(file)

    # Check statistic - How to test a number with training net
    img = numpy.random.randint(0,10000) # pick random feature in the test dataset    
    prediction = net.feedForward(testing_data[img][0]) #[0] is the 28x28 pixels
    print(f"Image number {img} in the testing set is a {testing_data[img][1]}, and the current network predicted a {numpy.argmax(prediction)}")
    
    figure, ax = plt.subplots(1, 2 , figsize=(8,4))

    print(testing_data[img][0])
    print(numpy.shape(testing_data[img][0]))

    ax[0].matshow(numpy.reshape(testing_data[img][0], (28,28)), cmap='gray') # color map
    ax[1].plot(prediction, lw = 2) # line width
    ax[1].set_aspect(10) 
    
    plt.show()

if __name__ == "__main__":
    main()