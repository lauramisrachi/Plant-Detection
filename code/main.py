#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:31:20 2018

@author: lauramisrachi
"""

from DataLoader import DataLoader
from PlantExtractor import PlantExtractor

""" Settings """

print('Setting parameters...')

data_path = '/Users/lauramisrachi/Documents/Bilberry/images'
images_names = ['image_1.png', 'image_2.png', 'image_3.png']
save_path = '.'
new_images_names = ['image_1_with_bb.png', 'image_2_with_bb.png', 
                    'image_3_with_bb.png']



""" Objects instantiation """


print('Instantiating the data loader and the plant extractor ...')

data_loader = DataLoader(data_path, images_names)
plant_extractor = PlantExtractor(save_path)


""" Loading dataset """

print('Loading the dataset ... ')

dataset = data_loader.load_images()


""" Detecting plants and saving images """

print('Detecting the plants in the images ...')

for i in range(len(images_names)):
    
    plant_extractor.save_img_with_bounding_box(dataset[i], new_images_names[i])

