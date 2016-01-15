# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:37:32 2016

@author: Hari
"""

import numpy as np
import inspect

class Network():
    
    def __init__(self, name, optimizer, cost_function):
        self.name = name
        self.layers = []
        self.optimizer = optimizer
        self.cost_function = cost_function
    
    def add_layer(self, layer):
        self.layers.append(layer)
        
        
    def backpropagation(self, x, y):
        # backpropagation
        activations = []
        weighted_inputs = []

        activation = x
        activations.append(activation)
        
        for layer in self.layers:
            weighted_input = np.dot(layer.weights, activation)
            weighted_inputs.append(weighted_input)
            activation = layer.function(weighted_input)
            activations.append(activation)
    
        print(self.weighted_inputs)
        print(self.activations)
        
    def train(self, training_x, training_y):
        # initialize weights
        input_layer_size = training_x[0].shape[0]
        self.layers[0].generate_weights(input_layer_size)
        if len(self.layers) > 1:
            for i, i_plus_1 in zip(self.layers, self.layers[1:]):
                i_plus_1.generate_weights(i.size)
            
        
        
        
    
    def __str__(self):
        opt_method = "Optimizer: " + self.optimizer + "\n"
        cost_func = "Cost Function: " + self.cost_function + "\n"
        architecture = " => ".join([layer.name for layer in self.layers]) + "\n\n"
        layer_list = "\n\n".join([str(layer) for layer in self.layers]) + "\n\n"
        return "~" + self.name + "~" + "\n" + opt_method + cost_func + "\n" + \
                   architecture + layer_list

class Layer():  
    def __init__(self, size, name):
        self.size = size
        self.name = name
        self.weights = None
    
    @staticmethod
    def function(x):
        return x
    
    @staticmethod
    def derivative(x):
        return 1
        
    def generate_weights(self, input_size):
        """ 
        ~ structure it as (l+1)th layer * lth layer matrix because we want
        inputs into current layer's nodes 
        """
        self.weights = np.random.randn(self.size, input_size) / np.sqrt(input_size)
    
    def __str__(self):
        return "Layer: %s \nNumber of nodes: %s" % (self.name, self.size)
         
         
class sigmoidLayer(Layer): 
    @staticmethod
    def function(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        return  (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
    
    def __str__(self):
        return super(sigmoidLayer, self).__str__() + "\nType: Sigmoid"


class tanhLayer(Layer):
    @staticmethod
    def function(x):
        return np.tanh(x)
    
    @staticmethod
    def derivative(x):
        return (1 + np.tanh(x / 2)) / 2
    
    def __str__(self):
        return super(tanhLayer, self).__str__() + "\nType: Tanh"
        
        
class softmaxLayer(Layer):
    """ Output layer only! """
    @staticmethod
    def function(x, all_x):
        return np.exp(x) / np.sum(np.exp(all_x))
    
    @staticmethod
    def derivative(x):
        pass
    
    def __str__(self):
        return super(softmaxLayer, self).__str__() + "\nType: Softmax"
        
class reluLayer(Layer): 
    @staticmethod
    def function(x):
        return np.max([0, x])
    
    @staticmethod
    def derivative(self, x):
        pass
    
    def __str__(self):
        return super(reluLayer, self).__str__() + "\nType: ReLU"
        
# tests
x = np.array([[1,0],[1,1],[0,1],[0,0]])  
y = np.array([1,1,0,1])  
        
network = Network("Test Network", "SGD", "MSE")
hidden_layer1 = sigmoidLayer(3, "Hidden Layer 1")
hidden_layer2 = sigmoidLayer(3, "Hidden Layer 2")
output_layer = tanhLayer(2, "Output Layer")

network.add_layer(hidden_layer1)
network.add_layer(hidden_layer2)
network.add_layer(output_layer)
network.train(x, y)

