from skimage import exposure, filters
import numpy as np

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
        image = filters.rank.maximum(image, np.ones((3, 3)))
        image = filters.rank.minimum(image, np.ones((3, 3)))
        return np.log(image+1)