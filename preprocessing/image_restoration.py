import sys
sys.path.append('../')

from imports import *

import numpy as np

class ImageRestorer:
    def __init__(self, method="median"):

        # method: 'mean_circular', 'mean_rectangular', 'median', 'gaussian', 'adaptive', 'wiener'

        self.method = method
    
    def process(self, image):
        if self.method == "mean_circular":
            # Mean filter
            image = self.mean_filter_using_circular_disk(image)
        elif self.method == "mean_rectangular":
            # Mean filter
            image = self.mean_filter_using_rectangular_disk(image)
        elif self.method == "median":
            # Median filter
            image = self.median_filter(image)
        elif self.method == "gaussian":
            # Gaussian filter
            image = self.gaussian_filter(image)
        elif self.method == "adaptive":
            # Adaptive filter
            image = self.adaptive_filter(image)
        elif self.method == "wiener":
            # Wiener filter
            image = self.wiener_filter(image)
        else:
            raise ValueError("Invalid method specified")
        
        return image
    
    def mean_filter_using_rectangular_disk(self, image, width=3, height=3):
        kernel = np.ones((height, width, 3)) / (height * width)
        filtered_image = convolve(image, kernel)
        return filtered_image


    def mean_filter_using_circular_disk(self, image, radius=3):
        # Define a disk structuring element
        selem = disk(radius)
        selem = selem[:, np.newaxis]
       
        # Apply the mean filter using the structuring element
        return rank.mean(image = image,footprint = selem)
    
    def median_filter(self, image):
        # Median filter
        return median(image)
    
    def gaussian_filter(self, image):
        # Gaussian filter
        return gaussian(image, sigma=1, mode='reflect', cval=0, multichannel=True, preserve_range=False, truncate=4.0)
    
    def adaptive_filter(self, image):
        # Adaptive filter
        return denoise_nl_means(image, h=0.8 * 1.0, fast_mode=True, patch_size=5, patch_distance=3, 
                                multichannel=True, preserve_range=False)
    
    def wiener_filter(self, image):
        # Wiener filter
        image = rgb2gray(image)
        psf = np.ones((5, 5)) / 25
        img = convolve2d(image, psf, 'same')
        rng = np.random.default_rng()
        img += 0.1 * img.std() * rng.standard_normal(image.shape)
        deconvolved_img = wiener(image, psf, 0.1)
        return deconvolved_img
