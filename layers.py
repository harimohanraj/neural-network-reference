# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:02:11 2016

@author: Hari
"""

import numpy as np

class Layer():  
    def __init__(self, size, name):
        self.size = size
        self.name = name
        self.weights = None
        self.bias = np.random.rand(1)
    
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
   