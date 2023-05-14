import sys
sys.path.append('../')

from imports import *

class ThresholdSegmentation:
    def init(self, method="Global", block_size=35, thresh_method="otsu", n_thresholds=3):
        self.method = method
        self.block_size = block_size
        self.thresh_method = thresh_method
        self.n_thresholds = n_thresholds
    
    def process(self, image):
        #TODO may need to do copy of the image
        # Convert the image to grayscale
        image = rgb2gray(image)
        
        if self.method == "Global":
            # Global thresholding
            thresh = threshold_otsu(image)
            binary = image > thresh
        elif self.method == "Otsu":
            # Otsu thresholding
            thresh = threshold_otsu(image)
            binary = image > thresh
        elif self.method == "Local":
            # Local adaptive thresholding
            binary = threshold_local(image, self.block_size)
        elif self.method == "Multilevel":
            # Multilevel thresholding
            if self.thresh_method == "otsu":
                thresh_func = threshold_otsu
            else:
                thresh_func = threshold_local
            # thresh_vals = threshold_multilevel(image, self.n_thresholds, method=thresh_func)
            # binary = np.digitize(image, thresh_vals)
        else:
            raise ValueError("Invalid method specified")
        
        return binary
