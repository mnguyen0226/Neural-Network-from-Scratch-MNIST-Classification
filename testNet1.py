"""
    Program that load trained model net1
    Command: python ./testNet1.py on terminal
"""
import pickle
import numpy
import matplotlib.pyplot as plt 
import matplotlib as mpl


import mnist_official_loader 
from mnist_official_loader import processData

def main():

    training_data, testing_data = mnist_official_loader.processData() # Data Preprocessing

    print("========First Run: [784,30,10] 30 epochs=========\n")
    
    # Load Trained Model
    pickleFile = "trained_models/trainedModel_784_30_10_30Epoch.pkl"
    with open(pickleFile, 'rb') as file:
        net = pickle.load(file)

    img = numpy.random.randint(0,10000) # pick random feature in the test dataset    
    prediction = net.feedForward(testing_data[img][0]) #[0] is the 28x28 pixels
    print(f"Image number {img} in the testing set is a {testing_data[img][1]}, and the current network predicted a {numpy.argmax(prediction)}")
    
    figure, axis = plt.subplots(1, 2 , figsize=(8,4))
    axis[0].matshow(numpy.reshape(testing_data[img][0], (28,28)), cmap='gray') # color map
    axis[1].plot(prediction, lw = 2) # line width
    axis[1].set_aspect(10) 
    
    plt.show()

if __name__ == "__main__":
    main()