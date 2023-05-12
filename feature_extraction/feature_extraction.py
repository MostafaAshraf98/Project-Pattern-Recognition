from skimage.feature import hog, local_binary_pattern
import cv2
import numpy as np
from skimage.color import rgb2gray
"""
# Example usage:
fe = FeatureExtractor()

# Load a set of images...
images = load_images()

# Extract HOG features...
hog_features = fe.extract_hog_features(images)

# Extract LBP features...
lbp_features = fe.extract_lbp_features(images)

# Extract SIFT features...
sift_features = fe.extract_sift_features(images)

# Extract SURF features...
surf_features = fe.extract_surf_features(images)

# Extract Fourier Descriptor features...
fourier_features = fe.extract_fourier_descriptor_features(images)

# Extract PCA features...
pca_features = fe.extract_pca_features(images)
"""

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_hog_features(self, images, hog_orientations=9,hog_pixels_per_cell=(8, 8), hog_cells_per_block=(2, 2)):
        hog_features = hog(images, orientations=hog_orientations, pixels_per_cell=hog_pixels_per_cell, cells_per_block=hog_cells_per_block,
                       visualize=False, multichannel=True)
        return hog_features

    def extract_lbp_features(self, images,lbp_num_points=8, lbp_radius=1):
        lbp_features = local_binary_pattern(images, lbp_num_points, lbp_radius)
        lbp_features = lbp_features.reshape(lbp_features.shape[0], -1)
        return lbp_features

    def extract_sift_features(self, images, sift_num_features=128):
        # sift = cv2.xfeatures2d.SIFT_create(128)
        sift = cv2.SIFT_create(sift_num_features)
        keypoints, sift_features = sift.detectAndComputeMulti(images, None)
        return sift_features

    def extract_surf_features(self, images, surf_num_features=64):
        surf = cv2.xfeatures2d.SURF_create(surf_num_features)
        keypoints, surf_features = surf.detectAndComputeMulti(images, None)
        return surf_features

    def extract_fourier_descriptor_features(self, images, num_coeffs=20):
        print(f'image shape: {images[0].shape}')
        contours = [max(cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
                    for image in images]
        contour_complexes = [np.empty(contour.shape[:-1], dtype=complex) for contour in contours]
        for i in range(len(contours)):
            contour_complexes[i].real, contour_complexes[i].imag = contours[i][:, 0, 0], contours[i][:, 0, 1]
        fourier_coeffs = [np.fft.fft(contour_complex)[:num_coeffs] for contour_complex in contour_complexes]
        fourier_coeffs = np.array(fourier_coeffs)
        return fourier_coeffs


# Fourier Descriptor
'''
The extract_fourier_descriptor_features function takes as input a NumPy array of images and an optional
parameter num_coeffs that specifies the number of Fourier coefficients to use as features (default is 20).

For each image, the function first converts it to grayscale and finds the contour with the largest area
using OpenCV's findContours function. It then converts the contour to a complex number representation 
and computes the Fourier coefficients of the contour using NumPy's fft function. 
Finally, the function extracts the first num_coeffs Fourier coefficients and appends them to a list of
Fourier descriptors.

The resulting list of Fourier descriptors is converted to a NumPy array and returned by the function. Note that Fourier Descriptor feature extraction can be computationally expensive, especially for large images or a large number of coefficients, so it may be necessary to optimize the implementation for performance.
'''