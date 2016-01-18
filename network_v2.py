# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:37:32 2016

@author: Hari
"""

import numpy as np
from layers import *


class Network():
    
    def __init__(self, name, optimizer, cost):
        self.name = name
        self.layers = []
        self.optimizer = optimizer
        self.cost = cost
        self.weights = None
        self.biases = None
        
        # self.dropout = True
        # self.regularization_type = "L2"
    
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def generate_weights_and_biases(self, training_x):
        input_size = training_x[0].shape[0]
        layer_sizes = [input_size] + [layer.size for layer in self.layers]
        weights = [np.random.randn(y, x) / np.sqrt(x) \
                        for x, y in zip(layer_sizes, layer_sizes[1:])]
        biases = [np.random.randn(1) for y in layer_sizes[1:]]       
        return weights, biases
        
    def backpropagation(self, x, y):
        # 0. Initialize empty arrays to hold gradients
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
    
        # 3. Compute output "error" (node delta) and gradients at output
        sigma_prime_of_wi = self.layers[-1].derivative(weighted_inputs[-1])
        error_delta = self.cost.derivative(y, activations[-1]) * sigma_prime_of_wi
        gradients_at_w[-1] = np.dot(error_delta, activations[-2].T)
        gradients_at_b[-1] = error_delta
      
        # 4. Backpropagate error (node deltas)
        for i in range(2,len(self.layers)):
            sigma_prime_of_wi = self.layers[-i].derivative(weighted_inputs[-i])
            error_delta = np.dot(self.layers[-i+1].weights.T, error_delta) * sigma_prime_of_wi
            gradients_at_w[-i] = np.dot(error_delta, activations[-i-1].T)
            gradients_at_b[-i]= error_delta
            
        # 5. Output gradient
        return gradients_at_w, gradients_at_b
    
    def train(self, training_x, training_y, iterations=10000):
        self.weights, self.biases = self.generate_weights_and_biases(training_x)
        for i in range(0,iterations):
            for 
    
    def __str__(self):
        opt_method = "Optimizer: " + self.optimizer + "\n"
        cost_func = "Cost Function: " + self.cost.__class__.__name__ + "\n"
        architecture = "Input Layer => " + " => ".join([layer.name for layer in self.layers]) + "\n\n"
        layer_list = "\n\n".join([str(layer) for layer in self.layers]) + "\n\n"
        return "~" + self.name + "~" + "\n" + opt_method + cost_func + "\n" + \
                   architecture + layer_list

     
# tests
x = np.array([[1,0],[1,1],[0,1],[0,0]])  
y = np.array([[1,0],[1,1],[0,1],[1,1]])  
        
network = Network(name="Test Network", \
                  optimizer="SGD", \
                  cost=MeanSquaredError())
hidden_layer1 = sigmoidLayer(3, "Hidden Layer 1")
output_layer = sigmoidLayer(2, "Output Layer")

network.add_layer(hidden_layer1)
network.add_layer(output_layer)
network.train(x, y)

