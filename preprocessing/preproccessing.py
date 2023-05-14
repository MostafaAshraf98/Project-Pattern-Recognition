import sys
sys.path.append('../')

from imports import *

class ImagePreprocessor:
    def __init__(self, method="HE"):
        # method: 'HE', 'AHE', 'CLAHE', 'log'
        self.method = method
        
    
    def process(self, image):
        if self.method == "HE":
            image = self._apply_histogram_equalization(image)
        elif self.method == "AHE":
            image = self._apply_adaptive_histogram_equalization(image)
        elif self.method == "CLAHE":
            image = self._apply_contrast_limited_adaptive_histogram_equalization(image)
        elif self.method == "log":
            image = self._apply_logarithmic_transformation(image)
        else:
            raise ValueError("Invalid method specified")
        
        return image
    
    def _apply_histogram_equalization(self, image):
        return exposure.equalize_hist(image)
    
    def _apply_adaptive_histogram_equalization(self, image):
        return exposure.equalize_adapthist(image, clip_limit=0.03)
    
    def _apply_contrast_limited_adaptive_histogram_equalization(self, image):
        return exposure.equalize_adapthist(image, clip_limit=0.03)
    
    def _apply_logarithmic_transformation(self, image):
        image = image.astype(np.float32) / 255.0  # Convert image to float and scale to [0, 1]
        log_img = np.log10(1 + image)  # Apply log transform to each channel
        log_img = (log_img / np.max(log_img)) * 255.0  # Scale log-transformed image back to [0, 255]
        log_img = log_img.astype(np.uint8)  # Convert back to uint8 format
        return log_img
    
        
