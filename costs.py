# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 21:17:19 2016

@author: Hari
"""

import numpy as np

class Cost():
    def __init__(self): 
        pass
    
    def function(y, a): 
        pass
    
    def derivative(y, a):
        pass
    
    
class MeanSquaredError(Cost):
    @staticmethod
    def function(y, a):
        return (1 / len(y)) * np.sum(np.power((y - a, 2)))
    
    @staticmethod
    def derivative(y, a):
        return a - y

class CrossCategoricalEntropyError(Cost):
    @staticmethod
    def function(y, a):
        return (-1 / len(y)) * np.sum(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))
    
    @staticmethod
    def derivative(y, a):
        return a - y
        

        
        
        