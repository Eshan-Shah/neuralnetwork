#Import in-built libraries
import math
import random

#Setup the class of a generic neuron
class Neuron:
    def __init__(self, weights, bias, inputs, activationFunction):
        self.weights = weights
        self.bias = bias
        self.inputs = inputs

        self.activationFunction = activationFunction
    
    def computeWeightedSum(self):
        weightedSum = 0
        for i in range(len(self.weights)):
            weightedSum += (self.weights[i] * self.inputs[i])

        weightedSum += self.bias
        return weightedSum
    
    def applyActivationFunction(self, z):
        return self.activationFunction(z)
    
    def computeLossFunction():
        pass

    def feedForward(self):
        z = self.computeWeightedSum()
        self.applyActivationFunction(z)

    def backPropogation(self):
        pass