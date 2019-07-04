#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 19:24:54 2018

@author: lauramisrachi
"""

import os

import numpy as np
from keras.preprocessing.image import load_img, img_to_array

from utils import get_images_filenames

class DataLoader(object):
    """ Class defining a data loader object to load the images as numpy arrays.
    
    INPUT:
        - data_path (str): path file were the data can be found.
    """
    
    def __init__(self, data_path):
        
        self.data_path = data_path
        self.images_path = get_images_filenames(self.data_path)
    
    
    def load_images(self):
        """ Load the images stored in the data_path folder as numpy arrays."""

        images = []

        for image_path in self.images_path:
            images.append(img_to_array(load_img(image_path)))

        images = np.asarray(images)
        print('The images were successfully loaded.')
        
        return images
