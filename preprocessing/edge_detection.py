import sys
sys.path.append('../')

from imports import *

class EdgeDetection:
    def __init__(self, method="sobel"):

        # method: 'sobel', 'prewitt', 'roberts', 'laplace', 'canny'
        # for roberts and canny: image must be 2D

        self.method = method
    
    def sobel(self, image):
        return sobel(image)
    
    def prewitt(self, image):
        return prewitt(image)
    
    def roberts(self, image):
        return roberts(rgb2gray(image))
    
    def laplace(self, image):
        return laplace(image)
    
    def canny(self, images, sigma=1.0, low_threshold=None, high_threshold=None, mask=None):
        if (len(images.shape) == 3):
            # Array of images
            edge_detected_imgs = np.zeros((images.shape[0], images.shape[1], images.shape[2]))
            for i in range(images.shape[0]):
                edge_detected_imgs[i] = canny(images[i], 
                                              sigma=sigma, low_threshold=low_threshold, 
                                              high_threshold=high_threshold, mask=mask)
            return edge_detected_imgs
        else:
            return canny(images, 
                         sigma=sigma, 
                         low_threshold=low_threshold, 
                         high_threshold=high_threshold, mask=mask)
    
    def process(self, image):
        if self.method == "sobel":
            # Sobel edge detection
            return self.sobel(image)
        elif self.method == "prewitt":
            # Prewitt edge detection
            return self.prewitt(image)
        elif self.method == "roberts":
            # Roberts edge detection
            return self.roberts(image)
        elif self.method == "laplace":
            # Laplacian edge detection
            return self.laplace(image)
        elif self.method == "canny":
            # Canny edge detection
            return self.canny(image)
        else:
            raise ValueError("Invalid method specified")
