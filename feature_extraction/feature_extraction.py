from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
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
    def __init__(self, hog_orientations=9, hog_pixels_per_cell=(8, 8), hog_cells_per_block=(2, 2),
                 lbp_num_points=8, lbp_radius=1, sift_num_features=128, surf_num_features=64,
                 num_fourier_coeffs=20, num_pca_components=20):
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.lbp_num_points = lbp_num_points
        self.lbp_radius = lbp_radius
        self.sift_num_features = sift_num_features
        self.surf_num_features = surf_num_features
        self.num_fourier_coeffs = num_fourier_coeffs
        self.num_pca_components = num_pca_components

    def extract_hog_features(self, images):
        hog_features = hog(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       visualize=False, multichannel=True)
        return hog_features

    def extract_lbp_features(self, images):
        lbp_features = local_binary_pattern(images, 8, 1)
        lbp_features = lbp_features.reshape(lbp_features.shape[0], -1)
        return lbp_features

    def extract_sift_features(self, images):
        # sift = cv2.xfeatures2d.SIFT_create(128)
        sift = cv2.SIFT_create(128)
        gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
        keypoints, sift_features = sift.detectAndComputeMulti(gray_images, None)
        return sift_features

    def extract_surf_features(self, images):
        surf = cv2.xfeatures2d.SURF_create(64)
        gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]
        keypoints, surf_features = surf.detectAndComputeMulti(gray_images, None)
        return surf_features

    def extract_fourier_descriptor_features(self, images, num_coeffs=20):
        print(f'image shape: {images[0].shape}')
        gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]
        contours = [max(cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
                    for gray_image in gray_images]
        contour_complexes = [np.empty(contour.shape[:-1], dtype=complex) for contour in contours]
        for i in range(len(contours)):
            contour_complexes[i].real, contour_complexes[i].imag = contours[i][:, 0, 0], contours[i][:, 0, 1]
        fourier_coeffs = [np.fft.fft(contour_complex)[:num_coeffs] for contour_complex in contour_complexes]
        fourier_coeffs = np.array(fourier_coeffs)
        return fourier_coeffs

    def extract_pca_features(self, images):
        """
        The extract_pca_features function takes as input a NumPy array of images and an optional parameter num_components that specifies the number of principal components to use as features (default is 20).
        For each image, the function flattens the image into a 1D vector and appends it to a list of image vectors. 
        It then converts the list of image vectors to a NumPy array and performs PCA using scikit-learn's PCA function. 
        Finally, the function extracts the first num_components principal components and returns them as the PCA features.
        """
        image_vectors = []
        for image in images:
            image_vectors.append(image.flatten())
        image_vectors = np.array(image_vectors)
        pca = PCA(n_components=self.num_pca_components)
        pca.fit(image_vectors)
        pca_features = pca.transform(image_vectors)
        return pca_features


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