"""
    ECE 4424 - Project Classify Image Using 2-layers Neural Network with MNIST data set
    Minh Nguyen
    12/7/2020
    *About Running: main() function that run real-time training and testing result(s)
     Note: This code is run on VSCode, in order to draw graph, we have to have #%%
     Related modules: 
        mnist_official_loader.py
        neuralNetwork.py
        mnistDB folder (download from MNIST site)

    *About pickled Testing trained model .py + trained_models folder (no need for running 3 models in main())
        testNet1.py => trainedModel_784_30_10_30Epoch.pkl
        testNet2.py => trainedModel_784_100_10_30Epoch.pkl
        testNet3.py => trainedModel_784_30_10_40Epoch.pkl

    # Save model - Don't use this unless you want to save the trained model in trained_models folder
    pickleFile = "trained_models/trainedModel_784_30_10_30Epoch.pkl"
    with open(pickleFile, 'wb') as file:
        pickle.dump(net1, file)
"""
#%%
import neuralNetwork 
from neuralNetwork import Network 
import psutil 
import mnist_official_loader 
from mnist_official_loader import processData
import time
import numpy
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pickle

def main():
    memory1 = psutil.virtual_memory().percent

    ########################################################### Data Preprocessing
    training_data, testing_data = mnist_official_loader.processData() # Data Preprocessing
    
    ########################################################### Running Net1 Real Time
    print("========First Run: [784,30,10] 30 epochs=========\n")

    net1 = neuralNetwork.Network([784,30,10])  # Create a 3 layers neural nets first layer 784 neurons, hidden layer 30 neurons and the last layers is 10 neurons
    net1.StochasticGD(training_data, testing_data, 30, 10, 3.0) # First run over 30 epochs, mini_batch_size = 10 and learning rate of 3

    memory2 = psutil.virtual_memory().percent
    memory_usage = abs(memory1 - memory2)
    print(f"The memory usage is: {memory_usage} bytes") 

    # Check statistic - How to test a number with trained net
    img1 = numpy.random.randint(0,10000) # pick random feature in the test dataset    
    prediction1 = net1.feedForward(testing_data[img1][0]) #[0] is the 28x28 pixels
    print(f"Image number {img1} in the testing set is a {testing_data[img1][1]}, and the current network predicted a {numpy.argmax(prediction1)}")
    
    figure1, ax1 = plt.subplots(1, 2 , figsize = (8,4))
    ax1[0].matshow(numpy.reshape(testing_data[img1][0], (28,28)), cmap='gray') # color map
    ax1[1].plot(prediction1, lw = 2) # line width
    ax1[1].set_aspect(10) 
    
    plt.show()
    
    ########################################################### Running Net2 Real Time
    # print("========Second Run: [784,100,10] 30 epochs=========\n")
    # net2 = neuralNetwork.Network([784,100,10])
    # net2.StochasticGD(training_data, testing_data, 30, 10, 3.0)    

    # memory2 = psutil.virtual_memory().percent
    # memory_usage = abs(memory1 - memory2)
    # print(f"The memory usage is: {memory_usage} bytes") 

    # # Check statistic - How to test a number with trained net
    # img2 = numpy.random.randint(0,10000) # pick random feature in the test dataset    
    # prediction2 = net2.feedForward(testing_data[img2][0]) #[0] is the 28x28 pixels
    # print(f"Image number {img2} in the testing set is a {testing_data[img2][1]}, and the current network predicted a {numpy.argmax(prediction2)}")
    
    # figure2, ax2 = plt.subplots(1, 2 , figsize = (8,4))
    # ax2[0].matshow(numpy.reshape(testing_data[img2][0], (28,28)), cmap='gray') # color map
    # ax2[1].plot(prediction2, lw = 2) # line width
    # ax2[1].set_aspect(10) 
    
    # plt.show()

    # ########################################################### Running Net3 Real Time
    # print("========Third Run: [784,30,10] 40 epochs=========\n")
    # net3 = neuralNetwork.Network([784,30,10])
    # net3.StochasticGD(training_data, testing_data, 40, 10, 3.0)    

    # memory2 = psutil.virtual_memory().percent
    # memory_usage = abs(memory1 - memory2)
    # print(f"The memory usage is: {memory_usage} bytes") 

    # # Check statistic - How to test a number with trained net
    # img3 = numpy.random.randint(0,10000) # pick random feature in the test dataset    
    # prediction3 = net3.feedForward(testing_data[img3][0]) #[0] is the 28x28 pixels
    # print(f"Image number {img3} is a {testing_data[img3][1]}, and the network predicted a {numpy.argmax(prediction3)}")
    
    # figure3, ax3 = plt.subplots(1, 2 , figsize = (8,4))
    # ax3[0].matshow(numpy.reshape(testing_data[img3][0], (28,28)), cmap='gray') # color map
    # ax3[1].plot(prediction3, lw = 2) # line width
    # ax3[1].set_aspect(10) 
    
    # plt.show()

    print("Finish Running")

###########################################################
if __name__ == "__main__":
    main()
