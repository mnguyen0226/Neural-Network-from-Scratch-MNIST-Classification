# Neural Network from Scratch MNIST Classification - ECE 4424
## How does this work? 
The network's job is to correctly classify pixel image numbers 0 - 9.

This 2 Layers Neural Networks is built from scratch with numpy package. The model is trained adn tested on Yann LeCun Dataset which can be found at http://yann.lecun.com/exdb/mnist/

The network is trained with Stochastic Gradient Descent.
## Preparation Steps after Git Clone the Repository:
**Step 1:** Make sure that your Python version is 3.7 or above.

**Step 2:** Import necessary Python package beside the Python Standard Lib: 
- (1) [psutil](https://pypi.org/project/psutil/) library
- (2) [matplotlib](https://matplotlib.org/3.1.1/faq/installing_faq.html) library 

**Step 3:** Download the dataset that I [downloaded](https://drive.google.com/drive/folders/1coms3ARgbH4-u5emWuJnPMZY0urCtk3M?usp=sharing) from Yann LeCunn MNIST website then put the four files ito the mnistDB Directory
## How do I run this neural network?
**Step 4:** Run main.py
```
python ./main.py
```
Wait for the network to be trained 30 epochs. Wait for a couple seconds until you see thee print statement "========First Run: [784,30,10] 30 epochs========="

Comment and uncomment code section in main.py to experience different neural network configurations.

## How do I test this neural network?
**Step 5:** Run testNet1.py, testNet2.py, testNet3.py
```
python testNet1.py
```
This command test the pretrained neural network model 1 with random input from the testing dataset. This pretrained model 1 is from "Running Net1 Real Time" code section in main.py

Do similar commands for testNet2.py, testNet3.py.
```
python testNet2.py
python testNet3.py
```
## Report:
Detailed Technical Instruction: "How To Run Neural Network.pdf"
## Honor Code @VT Warning:
You know what VT's Honor Code is, Hokies :). Don't do it. You have been warned.

## Paper:
Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.