#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 19:30:13 2018

@author: lauramisrachi
"""

import cv2
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
from keras import backend as K 
import os
import time

class PlantExtractor(object):
    """
    Class defining a plant extractor object that aims at detecting plants 
    and label them using bounding boxes. The algorithm uses the ExG-ExR vegetation
    index presented in the README file.

    INPUTS : 
        - save_path (str): path to the saving directory.
        - binary_threshold (float): the threshold used to define a region containing plant
        VS the opposite.
        - erosion_kernel_size (tuple): size of the kernel used for the erosion process
        applied to the binary image
        - dilatation_kernel_size (tuple): size of the kernel used for the dilatation
        process applied to the binary image
        - min_bounding_box_size (int) : minimum width or height of the labelling
        bounding box.
    """

    def __init__(self,
                 save_path,
                 binary_threshold=0,
                 erosion_kernel_size=(2, 2),
                 dilatation_kernel_size=(20, 20),
                 min_bounding_box_size=15):
        
        self.save_path = save_path
        self.result_folder = os.path.join(save_path, "results")
        self.binary_threshold = binary_threshold
        self.erosion_kernel_size = erosion_kernel_size
        self.dilatation_kernel_size = dilatation_kernel_size
        self.min_bounding_box_size = min_bounding_box_size

        self.create_results_folder()

    def create_results_folder(self):
        """ Create the results folder. """

        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)

    def compute_ExGExR_img(self, img):
        """ Computes the ExG-ExR version of an input image.
        
        INPUT :
            - img (np.ndarray) : a 3-channel image (RGB)
        
        OUTPUT : 
            - (np.ndarray): 1 channel grayscale image
        """
        
        r_star = img[:, :, 0] / 255
        g_star = img[:, :, 1] / 255
        b_star = img[:, :, 2] / 255
        
        sum_star = r_star + g_star + b_star + K.epsilon()
        r = r_star / sum_star
        g = g_star / sum_star
        b = b_star / sum_star
        
        return 3 * g - 2.4 * r - b
              
    def apply_threshold(self, img):
        """ Apply threshold to build a binary image.
        
        INPUT : 
            - img (np.ndarray) : input grayscale image
            
        OUTPUT : 
            thresholded (np.ndarray) : binary version of the image.
        """
        
        _, thresholded = cv2.threshold(img, self.binary_threshold, 
                                       255, cv2.THRESH_BINARY)

        return thresholded

    def apply_erosion_and_dilatation(self, img):
        """Apply erosion (to remove noise and keep region of interest) and dilatation
        (further highlight the rois) to the input image.
        
        INPUT : 
            - img : grayscale image 
            
        OUTPUT : 
            - dilated : grayscale image to which erosion and dilatation processes
            has been applied.
        """
        
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                   self.erosion_kernel_size)
        
        dilatation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                      self.dilatation_kernel_size)
        
        erosion = cv2.erode(img, erosion_kernel, iterations=1)
        dilated = cv2.dilate(erosion, dilatation_kernel, iterations=2)
        
        return dilated

    def get_bounding_boxes(self, img):
        """ Function that returns the plant-region labels using bounding boxes.
        Only the ROI with a width or height superior to min_bounding_box_size
        are considered of interest.
        
        INPUT : 
            - img : the initial img (as a grayscale image)
            
        OUTPUT : 
            - contour_list: a list containing element with the following shape:
                [x, y, w, h] providing the coordinates of the bounding boxes.
        """

        imgMod = self.compute_ExGExR_img(img)
        imgMod = self.apply_threshold(imgMod)
        imgMod = self.apply_erosion_and_dilatation(imgMod)
        imgMod = imgMod.astype('uint8')
        
        
        contours, _ = cv2.findContours(imgMod.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        contour_list = []
        
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            
            if w > self.min_bounding_box_size or h > self.min_bounding_box_size:
                contour_list.append([x, y, w, h])
                
        return contour_list

    def save_img_with_bounding_box(self, img, new_img_name):
        """ Successively apply the steps for plant detection on the input image
        (binary threshold, erosion and dilatation, contours finding). Save the newly
        generated image under its provided filename.

        INPUTS : 
            - img (np.ndarray): 3-channel input image (RGB)
            - new_img_name (str): name of the newly generated image (with bounding boxes)
        """
        
        print('Plants are being detected ...')
        start = time.time()
        contour_list = self.get_bounding_boxes(img)
        
        for contour in contour_list:
            x, y, w, h = contour
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
        end = time.time()
        timing = end - start
        
        print('The plants have been detected in %f seconds.' % (timing))
        
        cv2.imwrite(os.path.join(self.result_folder, new_img_name), img)
        
        print('The image with plant-region bounding boxes has been saved.')