#Setup the class of a generic neuron
class Neuron:
    def __init__(self, weights, bias, inputs, 
                 activationFunction, activationFunctionDerivative, 
                 lossFunction, lossFunctionDerivative, 
                 learningRate):
        
        self.weights = weights
        self.bias = bias
        self.inputs = inputs

        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative 
        self.lossFunction = lossFunction
        self.lossFunctionDerivative = lossFunctionDerivative

        self.learningRate = learningRate
    
    def computeWeightedSum(self):
        weightedSum = 0
        for i in range(len(self.weights)):
            weightedSum += (self.weights[i] * self.inputs[i])

        weightedSum += self.bias
        return weightedSum
    
    def applyActivationFunction(self, z):
        return self.activationFunction(z)
    
    def computeLossFunction(self, actualOutput, desiredOutput):
        return self.lossFunction(desiredOutput, actualOutput)
    
    def computeLossFunctionDerivative(self, actualOutput, desiredOutput):
        return self.lossFunctionDerivative(actualOutput, desiredOutput)

    def applyActivationFunctionDerivative(self, z):
        return self.activationFunctionDerivative(z)

    def combineForErrorSignals(self, lossFunctionDerivative, activationFunctionDerivative):
        return (lossFunctionDerivative * activationFunctionDerivative)
    
    def computeWeightGradient(self, delta, weight):
        return (delta * weight)

    def computeBiasGradient(self, delta):
        return delta
    
    def updateWeights(self, weightGradients):
        return [self.weights[i] - (self.learningRate * weightGradients[i]) for i in range(len(self.weights))]

    def updateBias(self, biasGradient):
        return (self.bias - (self.learningRate * biasGradient))

    def feedForward(self):
        self.weightedSum = self.computeWeightedSum()
        self.actualOutput = self.applyActivationFunction(self.weightedSum)

    def backPropogation(self):
        lossFunctionDerivative = self.computeLossFunctionDerivative()
        activationFunctionDerivative = self.applyActivationFunctionDerivative()

        delta = self.combineForErrorSignals(lossFunctionDerivative, activationFunctionDerivative)

        weightGradients = [self.computeWeightGradient(delta, weight) for weight in self.weights]
        biasGradient = self.computeBiasGradient(delta, self.bias)

        newWeights = self.updateWeights(weightGradients)
        newBias = self.updateBias(biasGradient)

