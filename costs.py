# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 21:17:19 2016

@author: Hari
"""

import numpy as np

class Cost():
    def __init__():
        pass
    
    def function(y, x):
        pass
    
    def derivative(y, x):
        pass
    
    
class MeanSquareError(Cost):
    # update this so it deals with multiple output nodes
    @staticmethod
    def function(y, x, fn):
        return np.power((y - fn(x)),2) / 2
    
    @staticmethod
    def derivative(y, x, fn):
        return fn(x) - y

class CrossCategoricalEntropyError(Cost):
    # update this so it deals with multiple output nodes
    @staticmethod
    def function(y, x, fn):
        return y
    
    @staticmethod
    def derivative(y, x, fn):
        return y
        
class LogLikelihoodError(Cost):
    # update this so it deals with multiple output nodes
     @staticmethod
    def function(y, x, fn):
        return y
    
    @staticmethod
    def derivative(y, x, fn):
        return y
        
        
        
        