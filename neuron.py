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
        return sum(self.weights[i] * self.inputs[i] for i in range(len(self.weights))) + self.bias
    
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
    
    def computeWeightGradient(self, delta, input):
        return (delta * input)

    def computeBiasGradient(self, delta):
        return delta
    
    def updateWeights(self, weightGradients):
        return [self.weights[i] - (self.learningRate * weightGradients[i]) for i in range(len(self.weights))]

    def updateBias(self, biasGradient):
        return (self.bias - (self.learningRate * biasGradient))

    def feedForward(self):
        weightedSum = self.computeWeightedSum()
        actualOutput = self.applyActivationFunction(weightedSum)

        return weightedSum, actualOutput

    def backPropogation(self, weightedSum, actualOutput, desiredOutput):
        lossFunctionDerivative = self.computeLossFunctionDerivative(actualOutput, desiredOutput)
        activationFunctionDerivative = self.applyActivationFunctionDerivative(weightedSum)

        delta = self.combineForErrorSignals(lossFunctionDerivative, activationFunctionDerivative)

        weightGradients = [self.computeWeightGradient(delta, input) for input in self.inputs]
        biasGradient = self.computeBiasGradient(delta)

        self.weights = self.updateWeights(weightGradients)
        self.bias = self.updateBias(biasGradient)







