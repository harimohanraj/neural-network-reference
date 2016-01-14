# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:37:32 2016

@author: Hari
"""

import numpy as np
import sklearn as sk
import inspect


class Layer():
    
    def __init__(self, size, name):
        self.size = size
        self.name = name
    
    def function(self, x):
        return x
    
    def derivative(self, x):
        return 1
        
    # static method? property? what is appropriate here?
    def generate_weights(self, input_size):
        """ 
        ~ Layer owns the weights that feed into it 
        ~ structure it as (l+1)th layer * lth layer matrix because we want
        inputs into current layer's nodes 
        ~ initialize as N(0,1/sqrt(# input nodes))
        """
        return np.random.randn(self.size, input_size)/np.sqrt(input_size)
              
class sigmoidLayer(Layer):
    
    def function(self, x):
        return 1/(1+np.exp(-x))
    
    def derivative(self, x):
        return self.function(x)*(1-self.function(x))


      
class tanhLayer(Layer):
    pass
class softmaxLayer(Layer):
    pass
class reluLayer(Layer):
    pass
        
        
        
    

