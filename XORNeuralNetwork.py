# APPLIED LINEAR ALGEBRA UTSA 
# FINAL PROJECT
# Neural Network
import numpy as np

# Configuration
learningRate = 0.1
epochs = 500
inputSize = 2
outputSize = 1
firstLayerNeuronNum = 20
secondLayerNeuronNum = 5
numOfTrainingData = 4

np.random.seed(22)

# Classes
# USES RELU AS THE ACTIVATION FUNCTION
class ActivationFunction:
    def __init__(self):
        pass
        
    def RelU(self, x):
        return(np.maximum(0, x))
    
    def RelUDerived(self, x):
        # Returns the derivative of the RelU
        return(np.where(x < 0, 0, 1)) 
    
# Two Hidden Layers 
class HiddenLayers:
    def __init__(self, firstLayerNeurons, secondLayerNeurons):
        self.firstLayerNerurons = firstLayerNeurons
        self.secondLayerNeurons = secondLayerNeurons

class NeuralNetwork:
    def __init__(self, inputSize, outputSize, firstLayerNeuronNum, secondLayerNeuronNum, activationFunction, learningRate):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.firstLayerNeuronNum = firstLayerNeuronNum
        self.secondLayerNeuronNum = secondLayerNeuronNum
        self.learningRate = learningRate
        self.activateNeuron = activationFunction

        # Weigths and biases
        self.W1 = np.random.randn(self.firstLayerNeuronNum, self.inputSize)
        self.W2 = np.random.randn(self.secondLayerNeuronNum, self.firstLayerNeuronNum)
        self.W3 = np.random.randn(self.outputSize, self.secondLayerNeuronNum)

        # print(f"Random W1: {self.W1}, W2: {self.W2}, W3: {self.W3}")
        self.B1 = np.zeros((self.firstLayerNeuronNum, 1))
        self.B2 = np.zeros((self.secondLayerNeuronNum, 1))
        self.B3 = np.zeros((self.outputSize, 1))

    # Forward propogation
    def forwardProp(self, inputs):
        # Forward 1
        self.Z1 = np.dot(self.W1, inputs) + self.B1
        # First Layer
        self.A1 = self.activateNeuron.RelU(self.Z1)
        # Forward 2
        self.Z2 = np.dot(self.W2, self.A1) + self.B2
        # Second Layer
        self.A2 = self.activateNeuron.RelU(self.Z2)
        # Prediction
        self.Z3 = np.dot(self.W3, self.A2) + self.B3
        self.A3 = self.Z3
        # print(f"Z1: {self.Z1}, A1: {self.A1}, Z2: {self.Z2}, A2: {self.A2}, Z3: {self.Z3}, A3: {self.A3}")
        return self.A3
    # Back propogation
    def backProp(self, input, expectedOutput):
        batchSize = input.shape[1]

        """
        MY NOTES:

        MSE - MEAN SQUARED ERROR = 1/(2 * batchSize) * (yExpected - yA3)^2
        A3 IS THE PREDICTION
        
        dA3/dA2 = d/dA2(W3 * A2 + B3) = W3
        dA2/dZ2 = 0 when Z2 < 0 and Z2 when Z2 >= 1
        dA3/dW3 =  A2
        dA3/dB3 = 1
        dZ2/dW2 = A1
        dZ2/dWB1 = 1
        dA1/dZ1 = 0 when Z1 < 0 and Z1 when Z1 >= 1
        dZ1/dW1 = input
        dZ1/dB1 = 1

        dMSE/dA3 = -1/batchSize(yExpected - yA3) or 1/batchSize(yA3 - yExpected)

        dMSE/dW3 = (dMSE/dA3) * (dA3/dW3)  
        dMSE/dB3 = (dMSE/dA3) * (dA3/dB3)

        dMSE/dW2 = (dMSE/dA3) * (dA3/dA2) * (dA2/dZ2) * (dZ2/dW2)
        dMSE/dB2 = (dMSE/dA3) * (dA3/dA2) * (dA2/dZ2) * (dZ2/dB2)

        dMSE/dW1 = (dMSE/dA3) * (dA3/dA2) * (dA2/dZ2) * (dZ2/dA1) * (dA1/dZ1) * (dZ1/dW1)
        dMSE/dB1 = (dMSE/dA3) * (dA3/dA2) * (dA2/dZ2) * (dZ2/dA1) * (dA1/dZ1) * (dZ1/dB1)
        """

        dZ3 = (1 / batchSize) * (self.Z3 - expectedOutput)

        dW3 = np.dot(dZ3, self.A2.T)
        dB3 = np.sum(dZ3, axis = 1, keepdims = True)

        dA2 = np.dot(self.W3.T, dZ3)
        dZ2 = dA2 * activationFunction.RelUDerived(self.Z2)
        dW2 = np.dot(dZ2, self.A1.T)
        dB2 = np.sum(dZ2, axis = 1, keepdims = True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * activationFunction.RelUDerived(self.Z1)
        dW1 = np.dot(dZ1, input.T)
        dB1 = np.sum(dZ1, axis = 1, keepdims = True)

        # Update weights and biases
        self.W3 -= self.learningRate * dW3
        self.B3 -= self.learningRate * dB3
        self.W2 -= self.learningRate * dW2
        self.B2 -= self.learningRate * dB2
        self.W1 -= self.learningRate * dW1
        self.B1 -= self.learningRate * dB1


    def train(self, inputs, expectedOutput, epochs):
        for epoch in range(epochs):
            prediction = self.forwardProp(inputs)
            loss = np.mean((prediction - expectedOutput)**2)
            self.backProp(inputs, expectedOutput)
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch: {epoch}, Loss: {loss:.5f}")

    def predict(self, input):
        return self.forwardProp(input)

# Creates an instance of the activationFunction
activationFunction = ActivationFunction()

# Initializes the layer
hiddenLayer = HiddenLayers(firstLayerNeuronNum, secondLayerNeuronNum)

# Creates an instance of the neural network
neuralNetwork = NeuralNetwork(inputSize, outputSize, firstLayerNeuronNum, secondLayerNeuronNum,activationFunction, learningRate)

# Traing Data XOR
# Project Data Set
projectInputs =  np.array([[0, 0, 1, 1], 
                           [0, 1, 0, 1]])
projectOutputs = np.array([[0, 1, 1, 0]])

# My test Inputs
myNumOfData = 6
myInputs =  np.array([[1, 0, 1, 0, 0, 1], 
                      [1, 1, 0, 1, 0, 0]])
myOutputs = np.array([[0, 1, 1, 1, 0, 1]])

print(f"Training with project data. Num Of Epochs: {epochs}")
neuralNetwork.train(projectInputs, projectOutputs, epochs)

print(f"DONE TRAINING!")


print("\nSample From Project Inputs")
for i in range(numOfTrainingData):
    inputSample = projectInputs[:, i:i + 1]
    
    outputSample = projectOutputs[:, i:i + 1] 
    prediction = neuralNetwork.predict(inputSample) 

    # Print the input, network prediction, and the true target value
    print(f"Input: {inputSample.T}, Prediction: {prediction[0, 0]:.4f}, Target: {outputSample[0, 0]}")

print("\nUsing My Inputs")
for i in range(myNumOfData):
    inputSample = myInputs[:, i:i + 1]
    
    outputSample = myOutputs[:, i:i + 1] 
    prediction = neuralNetwork.predict(inputSample) 
    
    # Print the input, network prediction, and the true target value
    print(f"Input: {inputSample.T}, Prediction: {prediction[0, 0]:.4f}, Target: {outputSample[0, 0]}")