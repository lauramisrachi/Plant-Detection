#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 19:24:54 2018

@author: lauramisrachi
"""

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

class DataLoader(object):
    
    """
    Class defining a data loader object to load the images as numpy arrays.
    
    INPUTS : 
        
        - data_path : path file were the data can be found.
        - images_names : list containing the names of the images available 
            in the data_path folder.
            
    
    
    """
    
    def __init__(self, data_path, images_names):
        
        self.data_path = data_path
        self.images_names = images_names
        self.images = []
    
    
    def load_images(self):
        
        
        """
        Function to load the images stored in the repository with path 'data_path'
        
        INPUTS : 
            - None
            
        OUTPUTS:
            - The list containing the images as numpy arrays.
        
        
        """
        
        for image_name in self.images_names:
            img_path = os.path.join(self.data_path, image_name)
            self.images.append(img_to_array(load_img(img_path)))
        
        self.images = np.asarray(self.images)
        
        print('The images were successfully loaded.')
        
        return self.images
