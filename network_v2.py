# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:37:32 2016

@author: Hari
"""

import numpy as np
from layers import *


class Network():
    
    def __init__(self, name, optimizer, cost_function):
        self.name = name
        self.layers = []
        self.optimizer = optimizer
        
        # replace with class and vectorize
        self.cost_function = lambda y,x: np.power((y - x), 2) / 2
        self.cost_function_derivative = lambda y,x: x - y
        
        # how do I implement these things?
        self.dropout = True
        self.regularization_type = "L2"
        
    
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def backpropagation(self, x, y):
        gradients_at_w = [np.zeros(layer.size) for layer in self.layers]
        gradients_at_b = [np.zeros(1) for layer in self.layers]
        
        # 1. Initialize first activation as training example  
        activations = []
        weighted_inputs = []
        activation = x
        
        # 2. Feedforward 
        for layer in self.layers:
            weighted_input = np.dot(layer.weights, activation)
            weighted_inputs.append(weighted_input)
            activation = layer.function(weighted_input)
            activations.append(activation)
        
        print(activations)
        print(weighted_inputs)
    
        # 3. Compute output "error" (node delta) and gradients at output
        sigma_prime_of_wi = self.layers[-1].derivative(weighted_inputs[-1])
        error_delta = self.cost_function_derivative(y, activations[-1]) * sigma_prime_of_wi
        gradients_at_w[-1] = np.dot(error_delta, activations[-2].T)
        
      
        # 4. Backpropagate error (node deltas)
        for i in range(2,len(self.layers)):
            sigma_prime_of_wi = self.layers[-i].derivative(weighted_inputs[-i])
            error_delta = np.dot(self.layers[-i+1].weights.T, error_delta) * sigma_prime_of_wi
            gradients_at_w[-i] = np.dot(error_delta, activations[-i-1].T)
            
        # 5. Output gradient
        return gradients_at_w
    
        
    def train(self, training_x, training_y):
        # initialize weights
        input_layer_size = training_x[0].shape[0]
        self.layers[0].generate_weights(input_layer_size)
        if len(self.layers) > 1:
            for i, i_plus_1 in zip(self.layers, self.layers[1:]):
                i_plus_1.generate_weights(i.size)
        
        # step through batch
            # compute gradient 
            # optimization to discern minimum
            # update weights
    
    def __str__(self):
        opt_method = "Optimizer: " + self.optimizer + "\n"
        cost_func = "Cost Function: " + self.cost_function + "\n"
        architecture = " => ".join([layer.name for layer in self.layers]) + "\n\n"
        layer_list = "\n\n".join([str(layer) for layer in self.layers]) + "\n\n"
        return "~" + self.name + "~" + "\n" + opt_method + cost_func + "\n" + \
                   architecture + layer_list

     
# tests
x = np.array([[1,0],[1,1],[0,1],[0,0]])  
y = np.array([[1,0],[1,1],[0,1],[1,1]])  
        
network = Network("Test Network", "SGD", "MSE")
hidden_layer1 = sigmoidLayer(3, "Hidden Layer 1")
output_layer = sigmoidLayer(2, "Output Layer")

network.add_layer(hidden_layer1)
network.add_layer(output_layer)
network.train(x, y)
network.backpropagation(np.array([[1],[0]]),np.array([[1],[1]]))

