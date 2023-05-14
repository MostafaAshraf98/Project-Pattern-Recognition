import sys
sys.path.append('../')

from imports import *

class RegionBasedSegmentation:
    def __init__(self, method="region_growing", **kwargs):

        # method: 'region_growing', 'region_splitting', 'region_merging'

        self.method = method
        self.kwargs = kwargs
    
    def process(self, image):
        # Convert the image to grayscale
        image = rgb2gray(image)
        
        if self.method == "region_growing":
            # Region Growing segmentation
            threshold = self.kwargs.get('threshold', 0.5)
            seed_point = self.kwargs.get('seed_point', None)
            if seed_point is None:
                seed_point = (image.shape[0]//2, image.shape[1]//2)
            mask = np.zeros_like(image)
            mask[seed_point] = 1
            segmented = ndi.binary_fill_holes(ndi.binary_dilation(mask, iterations=2))
            return segmented
        elif self.method == "region_splitting":
            # Region Splitting segmentation
            # min_size = self.kwargs.get('min_size', 50)
            # max_size = self.kwargs.get('max_size', 1000)
            segmented = felzenszwalb(image, scale=70, sigma=0.5)
            return segmented
        elif self.method == "region_merging":
            # Region Merging segmentation
            method = self.kwargs.get('method', "slic")
            if method == "slic":
                segmented = slic(image, n_segments=100, compactness=10, sigma=1)
            elif method == "quickshift":
                segmented = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
            else:
                raise ValueError("Invalid method specified")
            return segmented
        else:
            raise ValueError("Invalid method specified")