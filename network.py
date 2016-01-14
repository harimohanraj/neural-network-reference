# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:29:57 2016

@author: Hari

An annotated implementation of neural networks in Python 3 to learn how these
goddamn things work.

"""


import numpy as np
import inspect


class Activation():
    def __init__(self, activation_type):
        self.activation_type = activation_type
        if activation_type == "sigmoid":
            self.activation_func = self.sigmoid
    
    def sigmoid(self, x, derivative=False):
        if derivative:
            return ((1/1+np.exp(-x)))*(1-((1/1+np.exp(-x))))
        else:
            return 1/(1+np.exp(-x))
    
    def __str__(self):
        function_source = inspect.getsource(self.activation_func)
        return "Activation Object \nType: %s \nSource:  \n %s" % \
        (self.activation_type, function_source)
    
    
class Layer():
    def __init__(self, name, size, activation_function="sigmoid"):
        self.name = name
        self.size = size
        self.activation = Activation("sigmoid")
    
    def __str__(self):
        return "Layer: %s \nNumber of nodes: %s \nActivation Type: %s" % \
        (self.name, self.size, self.activation.activation_type)


class Network():
    def __init__(self, name): 
        self.name = name
        self.layers = []
        self.weights = []
        self.biases = []

    def add_layer(self, layer):
        """Add new layer to neural network"""
        self.layers.append(layer)
        if len(self.layers) > 1: # only generate weights for non-input layers
            self.weights.append(self.generate_weights())
            self.biases.append(np.random.rand())

    def generate_weights(self):
        """Look at two most recent layers added and generate weights for 
        connections between them"""
        return np.random.rand(self.layers[-1].size, self.layers[-2].size)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return ((1/1+np.exp(-x)))*(1-((1/1+np.exp(-x))))
        else:
            return 1/(1+np.exp(-x))
    
    def backpropagation(self, x, y):
        # initialize 
        activations = []
        weighted_inputs = []

        # feedforward
        activation = x
        weights_and_biases = zip(self.weights, self.biases)
        activations.append(activation)
        
        for w,b in weights_and_biases:
            weighted_input = np.dot(w, activation) + b
            activation = self.sigmoid(weighted_input) 
            # need to replace with layer-specific activation
            
            activations.append(activation)
            weighted_inputs.append(weighted_input)
        
        # backward pass
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        
        delta = (activations[-1] - y) * \
                self.sigmoid(activations[-1], derivative=True)
                # need to replace with layer-specific activation
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, len(self.layers)):
            delta = np.dot(self.weights[-l].T, delta) * \
                    self.sigmoid(weighted_inputs[-l], derivative=True)
            gradient_w[-l] = np.dot(activations[-l], delta)
            gradient_b[-l] = delta
            
            
    
    def __str__(self):
        architecture = " => ".join([layer.name for layer in self.layers]) + "\n\n"
        layer_list = "\n\n".join([str(layer) for layer in self.layers]) + "\n\n"
        return "~" + self.name + "~" + "\n\n" + architecture + layer_list
     
     
class Optimizer():
    def __init__(self, type):
        self.type = type

    def gradient_descent():
        pass
    def stochastic_gradient_descent():
        pass
    def mb_stochastic_gradient_descent():
        pass
    
    def __str__(self):
        pass
         
# Setup network
net1 = Network("Test Network 1")
input1 = Layer("Input", 2)
hidden1 = Layer("Hidden", 2)
output1 = Layer("Output", 2)

# Add some layers
net1.add_layer(input1)
net1.add_layer(hidden1)
net1.add_layer(output1)
print(str(net1))


