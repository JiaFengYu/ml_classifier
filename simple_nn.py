from classifier import classifier, argmax
from random import random, seed, shuffle
from math import exp

class Unit(object):
    '''Class that represents a 'neuron' in the ANN.'''
    def __init__(self, n):
        '''Allocate n+1 weights for the neural unit.'''
        self.weights = [random() for _ in range(n + 1)] # random weights.
        self.delta = 0          # error in the output.
        self.output = 0         # computed output (weights x inputs)

    @staticmethod
    def sigmoid(x):
        '''The classic 'sigmoid' activation function.'''
        return 1.0 / (1.0 + exp(-x))

    @staticmethod
    def derivative(y):
        '''The derivative of the sigmoid function, given an output 'y'.'''
        return y * (1.0 - y)

    def activate(self, inputs):
        '''Compute the activation for a unit.'''
        activation = sum(w * x for w, x in zip(self.weights, inputs + [1.0]))
        self.output = Unit.sigmoid(activation)
        return self.output
    
class BPNN(classifier):
    '''Classic 3-layer artificial neural network, trained with
    back-propagation.

    To keep it relatively simple, each layer is fully connected. Each
    of the inputs are connected to all of the hidden units, and each
    of the hidden units are connected to all of the output units.
    
    There are really only two 'public' methods:
        train(self, training_data)
        predict(self, data_point)
    '''
    def __init__(self, n_input, n_hidden, n_output):
        '''Create the hidden and output layers.'''
        hidden_layer = [Unit(n_input) for _ in range(n_hidden)]
        output_layer = [Unit(n_hidden) for _ in range(n_output)]
        self.network = [hidden_layer, output_layer]

    def forward_propagate(self, inputs):
        '''Evaluate the inputs.'''
        for layer in self.network:
            # Compute the activation of this layer.
            next_inputs = [unit.activate(inputs) for unit in layer]

            # Use it to compute the activation of the next layer.
            inputs = next_inputs
        return inputs
    
    def backward_propagate(self, outputs):
        '''Propagate error signal backwards through the network,
        given the correct 'outputs'.'''
        next_layer = None
        for layer in reversed(self.network):
            if next_layer:
                # Not at the output - get signal from next layer.
                for j, unit in enumerate(layer):
                    error = sum(x.weights[j] * x.delta for x in next_layer)
                    unit.delta = error * Unit.derivative(unit.output)
            else:
                # At the output - error signal is the difference between
                # the predicted and expected outputs.
                for unit, output in zip(layer, outputs):
                    error = unit.output - output
                    unit.delta = error * Unit.derivative(unit.output)

            next_layer = layer
                        
    def update_weights(self, inputs, eta):
        '''Update the weights in the network.'''
        inputs = inputs + [1.0] # Append the bias input.
        for layer in self.network:
            n = len(inputs)
            for unit in layer:
                for j in range(n):
                    unit.weights[j] -= eta * unit.delta * inputs[j]
            # Outputs from this layer become the inputs of the next.
            inputs = [unit.output for unit in layer]

    def train(self, train_data, n_iterations = 1000, eta = 0.2):
        '''Train the network with the specified training data
        and learning rate 'eta'.'''
        #
        # Train repeatedly using the entire training set, for the
        # specified number of iterations. An 'iteration' is often
        # called an 'epoch' in ANN literature.
        #
        n_outputs = len(self.network[-1])
        for iteration in range(n_iterations):
            total_error = 0
            # For every item in the training set.
            for item in train_data:
                # Get the computed outputs.
                predicted_outputs = self.forward_propagate(item.data)

                # Get the correct outputs.
                correct_outputs = [0.0] * n_outputs
                correct_outputs[item.label] = 1.0

                # Accumulate the total output error.
                total_error += sum((c - p) ** 2 for c, p in
                                   zip(correct_outputs, predicted_outputs))

                # Now train the network.
                self.backward_propagate(correct_outputs)
                self.update_weights(item.data, eta)

            if iteration % 100 == 0:
                print("iteration={}, error={:.3f}".format(iteration, total_error))

    def predict(self, data_point):
        '''Compute the predicted output for this data point.'''
        outputs = self.forward_propagate(data_point)
        return argmax(outputs)


if __name__ == "__main__":
    seed(1)
    from datasets import *
    from classifier import evaluate, normalize_dataset
    iris_dataset = read_iris_dataset()
    seeds_dataset = read_seeds_dataset()
    wine_dataset = read_wine_dataset()

    iris_dataset = normalize_dataset(iris_dataset)
    seeds_dataset = normalize_dataset(seeds_dataset)
    wine_dataset = normalize_dataset(wine_dataset)

    shuffle(iris_dataset)
    shuffle(seeds_dataset)
    shuffle(wine_dataset)

    print("The iris dataset.")
    print(evaluate(iris_dataset, BPNN, 4, n_input=4, n_hidden=4, n_output=3))

    print("The seeds dataset.")
    print(evaluate(seeds_dataset, BPNN, 4, n_input=7, n_hidden=7, n_output=3))

    print("The wine dataset.")
    print(evaluate(wine_dataset, BPNN, 4, n_input=13, n_hidden=13, n_output=3))

    print("The xor dataset.")
    xor = [data_item(0, [0.0, 0.0]),
           data_item(1, [1.0, 0.0]),
           data_item(1, [0.0, 1.0]),
           data_item(0, [1.0, 1.0])]

    nn = BPNN(2, 3, 2)
    nn.train(xor, 2000)
    n_correct = 0
    for item in xor:
        if nn.predict(item.data) == item.label:
            n_correct += 1
    print("accuracy: ", n_correct / len(xor))
