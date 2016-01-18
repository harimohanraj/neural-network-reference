# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 00:58:15 2016

@author: Hari
"""

import numpy as np

class GradientDescent():
    
    def __init__(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def run(self, training_x, training_y, iterations):
        # shuffle data if online sgd
        data = np.concatenate((training_x, training_y), axis=1)
        if self.batch_size == 1: np.random.shuffle(data)
        batch = data[i::self.batch_size]    
        
        # iterate through batch
        for record in batch:
            pass
            