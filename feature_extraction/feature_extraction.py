from skimage.feature import hog, local_binary_pattern
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import daisy
import matplotlib.pyplot as plt
from scipy.fftpack import fft


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

    def extract_hog_features(self, images, hog_orientations=8, 
                             hog_pixels_per_cell=(8, 8), 
                             hog_cells_per_block=(2, 2)):
        if(len(images.shape) == 3):
            #Array of images
            hog_features = []
            for i in range(images.shape[0]):
                hog_feature = hog(images[i], 
                                orientations=hog_orientations,
                                pixels_per_cell=hog_pixels_per_cell,
                                cells_per_block=hog_cells_per_block,
                                channel_axis = None)
                hog_features.append(hog_feature)
        else:
            #Single image
            hog_features = hog( images, 
                                orientations=hog_orientations,
                                pixels_per_cell=hog_pixels_per_cell,
                                cells_per_block=hog_cells_per_block,
                                channel_axis = None)
        return np.array(hog_features)
    
    def extract_lbp_features(self, images,lbp_num_points=8, lbp_radius=1):
        if(len(images.shape) == 3):
            lbp_features = []
            for i in range(images.shape[0]):
                feature = local_binary_pattern(images[i], lbp_num_points, lbp_radius)
                feature = feature.flatten()
                lbp_features.append(feature)
        else:
            lbp_features = local_binary_pattern(images, lbp_num_points, lbp_radius)
            lbp_features = lbp_features.flatten()
        return np.array(lbp_features)

    def extract_sift_features(self, images, sift_num_features=128):
        # sift = cv2.xfeatures2d.SIFT_create(128)
        print(f'Images: {images.shape}')
        sift = cv2.SIFT_create(nfeatures=sift_num_features)
        keypoints = []
        sift_features = []
        # if it is a single image
        if(len(images.shape) == 2):
            k, s = sift.detectAndCompute(images, mask = None)
            return np.array(s)
        
        max_length = -1
        for i in range(images.shape[0]):
            _, s = sift.detectAndCompute(images[i], mask = None)
            
            if (s is None):
                print(i)
                print(images[i].shape)
                plt.imshow(images[i])
                plt.show()
            s = s.flatten()
            sift_features.append(s)
            if len(s) > max_length:
                max_length = len(s)


        for i in range(len(sift_features)):
            if len(sift_features[i]) < max_length:
                sift_features[i] = np.pad(sift_features[i], (0, max_length - sift_features[i].shape[0]), 'constant')
        # sift_features = np.array(sift_features)
        # sift_features = sift_features.reshape(sift_features.shape[0], -1)
        return sift_features
    
    def extract_daisy_features(self, images):
        descs_features = []
        
        if len(images.shape) == 2:
            #Single Image
            descs = daisy(images, step=180, radius=58, rings=2, 
                          histograms=6, orientations=8, visualize=False)
            descs = descs.flatten()
            return descs
        
        for image in images:
            descs = daisy(image, step=180, radius=58, rings=2, 
                          histograms=6, orientations=8, visualize=False)
            descs = descs.flatten()
            descs_features.append(descs)
        descs_features = np.array(descs_features)
        return descs_features
    

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
    
    def extract_orb_features(self, images, features=100):
        # print(f'images shape: {images.shape}')
        
        descriptors = []
        max_length = -1
        if (len(images.shape) == 3):
            #Array of images
            for i in range(images.shape[0]):
                image = images[i]
                # Create an ORB object with specified parameters
                orb = cv2.ORB_create(nfeatures=features, scaleFactor=1.2, nlevels=8)

                # Detect keypoints in the image
                keypoints = orb.detect(image, None)

                # Compute descriptors for the keypoints
                keypoints, descriptor = orb.compute(image, keypoints)
                descriptor = descriptor.flatten()

                descriptors.append(descriptor)

                if len(descriptor) > max_length:
                    max_length = len(descriptor)

            for i in range(len(descriptors)):
                if len(descriptors[i]) < max_length:
                    descriptors[i] = np.pad(descriptors[i], (0, max_length - descriptors[i].shape[0]), 'constant')
                


        else:
            #Single image
            # print('Single image')
            # Create an ORB object with specified parameters
            orb = cv2.ORB_create(nfeatures=features, scaleFactor=1.2, nlevels=8)

            # Detect keypoints in the image
            keypoints = orb.detect(images, None)

            # Compute descriptors for the keypoints
            keypoints, descriptors = orb.compute(images, keypoints)
            descriptors = descriptors.flatten()

        # Return the descriptors as a numpy array
        return np.array(descriptors)

    def RI_HOG(image, cell_size=(8, 8), block_size=(2, 2), nbins=9, radius=1, neighbors=8):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradient magnitude and orientation
        grad_mag = cv2.Sobel(gray, cv2.CV_32F, 1, 1)
        # grad_orient = cv2.phase(grad_x, grad_y, angleInDegrees=True)

        # Compute HOG features
        hog = cv2.HOGDescriptor(_winSize=(gray.shape[1] // cell_size[1] * cell_size[1], gray.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        hog_feat = hog.compute(gray, winStride=(cell_size[1], cell_size[0]))

        # Compute CLBP features
        clbp = cv2.LBP(radius=radius, neighbors=neighbors)
        clbp_feat = clbp.compute(gray)

        # Concatenate HOG and CLBP features
        features = np.concatenate((hog_feat, clbp_feat), axis=1)

        return features
    
    def extract_hu_moments_features(self, images):
        if (len(images.shape) == 3):
            #Array of images
            hu_moments_list = []
            for i in range(images.shape[0]):
                # Find contours in the binary image
                contours,  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Select the largest contour
                contour = max(contours, key=cv2.contourArea)

                # Calculate Hu moments
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments)

                # Log transform Hu moments to make them scale invariant
                hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))

                # Print Hu moments
                hu_moments_list.append(hu_moments.flatten())

            res = np.array(hu_moments_list)
            return res
        
        else:
            # Find contours in the binary image
            contours,  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Select the largest contour
            contour = max(contours, key=cv2.contourArea)

            # Calculate Hu moments
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments)

            # Log transform Hu moments to make them scale invariant
            hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))

            # Print Hu moments
            return hu_moments.flatten()
        

    def extract_convex_hull_features(self, images, max_length_train, test=False):
        if(len(images.shape) == 3):
            max_length = -1
            features = []
            for image in images:
                # Binarize the image

                # Find contours in the image
                contours,_  = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the convex hull of the largest contour
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(largest_contour)
                    if (len(hull.flatten()) > max_length):
                        max_length = len(hull.flatten())
                    features.append(hull.flatten())
                else:
                    # If there are no contours, append an array of zeros to the feature list
                    features.append(np.zeros(2))

            if(test):
                if (max_length < max_length_train):
                    max_length = max_length_train

            for i in range(len(features)):
                if len(features[i]) < max_length:
                    features[i] = np.pad(features[i], (0, max_length - features[i].shape[0]), 'constant')

            return np.array(features), max_length
        

        else:
            print('Single')
            # Find contours in the image
            contours,  = cv2.findContours(images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the convex hull of the largest contour
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)
                features.append(hull.flatten())
            else:
                # If there are no contours, append an array of zeros to the feature list
                features.append(np.zeros(2))
            return np.array(features[0])
            

    def elliptical_fourier_descriptors(self, imgs):
        """
        Calculate the elliptical Fourier feature vector for each image in the input array.

        Args:
            imgs (numpy.ndarray): An array of grayscale images. Each image should be a 2D numpy array.

        Returns:
            numpy.ndarray: An array of elliptical Fourier feature vectors. Each row of the array is a feature vector for an image.
        """

        # Define the number of Fourier coefficients to calculate.
        n_coeffs = 20

        # Define the number of points to sample on each contour.
        n_samples = 200

        # Define the indices of the Fourier coefficients to keep.
        coeffs_to_keep = range(1, n_coeffs + 1)

        # Define the output array.
        efds = np.zeros((len(imgs), n_coeffs * 4))

        for i, img in enumerate(imgs):
            # Find the contour of the image.
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Sample the contour.
            contour = contours[0][:, 0, :]
            contour = self.resample_contour(contour, n_samples)

            # Calculate the complex Fourier coefficients of the contour.
            fourier_coeffs = fft(contour[:, 0] + 1j * contour[:, 1])

            # Take the first n_coeffs coefficients.
            fourier_coeffs = fourier_coeffs[coeffs_to_keep]

            # Calculate the elliptical Fourier descriptors.
            a0 = np.real(fourier_coeffs[0]) / n_samples
            b_coeffs = -np.imag(fourier_coeffs[1:]) / n_samples
            a_coeffs = np.real(fourier_coeffs[1:]) / n_samples
            efds[i] = np.concatenate(([a0], np.ravel(a_coeffs), np.ravel(b_coeffs)))

        return efds
    def resample_contour(self, contour, n_samples):
        """
        Resample a contour to have a specified number of points.

        Args:
            contour (numpy.ndarray): A 2D array containing the (x, y) coordinates of the contour points.
            n_samples (int): The number of points to sample on the contour.

        Returns:
            numpy.ndarray: A 2D array containing the resampled (x, y) coordinates of the contour points.
        """

        # Calculate the arc length of the contour.
        arc_length = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
        arc_length = np.insert(arc_length, 0, 0) / arc_length[-1]

        # Create a uniformly spaced grid of points along the arc length.
        t = np.linspace(0, 1, n_samples)

        # Interpolate the contour points along the arc length.
        x = np.interp(t, arc_length, contour[:, 0])
        y = np.interp(t, arc_length, contour[:, 1])

        return np.column_stack((x, y))


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