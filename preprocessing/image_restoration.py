from skimage.filters import median, gaussian, wiener
from skimage.restoration import denoise_nl_means, wiener
from skimage.filters import rank
from skimage.morphology import disk
from skimage.draw import rectangle

class ImageRestorer:
    def __init__(self, method="median"):
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
    
    def mean_filter_using_rectangular_disk(image, width=3, height=3):
        # Define a rectangular structuring element
        selem = rectangle(image.shape, width=width, height=height)
        
        # Apply the mean filter using the structuring element
        return rank.mean(image, selem)

    def mean_filter_using_circular_disk(image, radius=3):
        # Define a disk structuring element
        selem = disk(radius)
        
        # Apply the mean filter using the structuring element
        return rank.mean(image, selem)
    
    def median_filter(self, image):
        # Median filter
        return median(image)
    
    def gaussian_filter(self, image):
        # Gaussian filter
        return gaussian(image, sigma=1, mode='reflect', cval=0, multichannel=None, preserve_range=False, truncate=4.0)
    
    def adaptive_filter(self, image):
        # Adaptive filter
        return denoise_nl_means(image, h=0.8 * 1.0, fast_mode=True, patch_size=5, patch_distance=3, 
                                multichannel=True, preserve_range=False, sigma=None)
    
    def wiener_filter(self, image):
        # Wiener filter
        return wiener(image, 1, noise=None, mask=None, 
                      psf=None, balance=0.1, clip=True, preserve_range=False)
