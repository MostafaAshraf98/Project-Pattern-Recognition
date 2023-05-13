from skimage.feature import hog, local_binary_pattern
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import daisy
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pyefd import elliptic_fourier_descriptors


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
        self.sift_max_length = -1
        


    def extract_hog_features(self, images, hog_orientations=8, 
                             hog_pixels_per_cell=(32, 16), 
                             hog_cells_per_block=(2, 1)):
        # Increasing size of hog_pixels_per_cell decreases size of o/p features
        # Increasing size of hog_cells_per_block increases size of o/p features
        #Array of images
        hog_features = []
        for i in range(images.shape[0]):
            hog_feature = hog(images[i], 
                            orientations=hog_orientations,
                            pixels_per_cell=hog_pixels_per_cell,
                            cells_per_block=hog_cells_per_block,
                            channel_axis = None)
            hog_features.append(hog_feature)

        return np.array(hog_features)
    
    def extract_lbp_features(self, images, lbp_num_points=8, lbp_radius=1):
        lbp_features = []
        for image in images:
            feature = local_binary_pattern(image, lbp_num_points, lbp_radius)
            feature = feature.flatten()
            lbp_features.append(feature)
        return np.array(lbp_features)

    def extract_sift_features(self, _images, sift_num_features=128):
        # sift = cv2.xfeatures2d.SIFT_create()
        images = np.copy(_images)
        images = images.astype(np.uint8)
        sift = cv2.SIFT_create(nfeatures=sift_num_features, nOctaveLayers=5)
        keypoints = []
        sift_features = []
        
        failed_images = []
        train_flag = False
        if (self.sift_max_length == -1):
            train_flag = True
        for i in range(images.shape[0]):
            # print(images[i].shape)
            # plt.imshow(images[i])
            # plt.show()
            keypoints, s = sift.detectAndCompute(images[i], mask = None)
            # print(i, type(s), end = ' ')
            if (s is None):
                # print('None')
                # print('Keypoints: ', keypoints)
                failed_images.append(i)
                sift_features.append(np.zeros((0,0)))
                continue
            # print(s.shape)
            s = s.flatten()
            sift_features.append(s)
            if (train_flag and len(s) > self.sift_max_length):
                self.sift_max_length = len(s)

        # Padding
        for i in range(len(sift_features)):
            if sift_features[i].shape[0] == 0:
                sift_features[i] = np.zeros(self.sift_max_length)
            elif sift_features[i].shape[0] < self.sift_max_length:
                sift_features[i] = np.pad(sift_features[i], (0, self.sift_max_length - sift_features[i].shape[0]), 'constant')
        sift_features = np.array(sift_features)
        return sift_features
    
    def extract_daisy_features(self, images):
        descs_features = []
        
        for image in images:
            descs = daisy(image, step=180, radius=58, rings=2, 
                          histograms=6, orientations=8, visualize=False)
            descs = descs.flatten()
            descs_features.append(descs)
        descs_features = np.array(descs_features)
        return descs_features
    
    def extract_fourier_descriptor_features(self, images, num_coeffs=20):
        
        contours = [max(cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
                    for image in images]
        contour_complexes = [np.empty(contour.shape[:-1], dtype=complex) for contour in contours]
        for i in range(len(contours)):
            contour_complexes[i].real, contour_complexes[i].imag = contours[i][:, 0, 0], contours[i][:, 0, 1]
        fourier_coeffs = [np.fft.fft(contour_complex)[:num_coeffs] for contour_complex in contour_complexes]
        fourier_coeffs = np.array(fourier_coeffs)
        return fourier_coeffs
    
    def extract_orb_features(self, images, features=100):
        
        descriptors = []
        max_length = -1
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

        # Return the descriptors as a numpy array
        return np.array(descriptors)

    def RI_HOG(self, images, cell_size=(8, 8), block_size=(2, 2), nbins=9, radius=1, neighbors=8):
        descriptors = []
        for image in images:

            # Compute gradient magnitude and orientation
            grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
            grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
            grad_mag, grad_orient = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

            # Convert grad_mag to CV_8U type
            grad_mag = cv2.convertScaleAbs(grad_mag)

            # Compute HOG features
            hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // cell_size[1] * cell_size[1], image.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)
            hog_feat = hog.compute(grad_mag, winStride=(cell_size[1], cell_size[0]))

            # Compute CLBP features
            clbp_feat  = self.extract_lbp_features(np.array([image]), lbp_radius=radius, lbp_num_points=neighbors)[0]
            
            # Concatenate HOG and CLBP features
            features = np.concatenate((hog_feat, clbp_feat))
            descriptors.append(features.flatten())

        descriptors = np.array(descriptors)
        return descriptors

    
    def extract_hu_moments_features(self, images):
        
        hu_moments_list = []
        for image in images:
            
            # Find contours in the binary image
            contours  = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Select the largest contour
            contour = max(contours[0], key=cv2.contourArea)

            # Calculate Hu moments
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments)

            # Log transform Hu moments to make them scale invariant
            hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))

            # Print Hu moments
            hu_moments_list.append(hu_moments.flatten())

        res = np.array(hu_moments_list)
        return res
    
    def extract_convex_hull_features(self, images, max_length_train=-1):
        features = []
        for image in images:
            # Find contours in the image
            contours,_  = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the convex hull of the largest contour
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)
                if (len(hull.flatten()) > max_length_train):
                    max_length_train = len(hull.flatten())
                features.append(hull.flatten())
            else:
                # If there are no contours, append an array of zeros to the feature list
                features.append(np.zeros(2))

        for i in range(len(features)):
            if len(features[i]) < max_length_train:
                features[i] = np.pad(features[i], (0, max_length_train - features[i].shape[0]), 'constant')

        return np.array(features), max_length_train

    def elliptical_fourier_descriptors(self, imgs):

        # Define the number of Fourier coefficients to calculate.
        n_coeffs = 20

        # Define the number of points to sample on each contour.
        n_samples = 200

        # Define the indices of the Fourier coefficients to keep.
        coeffs_to_keep = range(1, 2*n_coeffs + 1)

        # Define the output array.
        efds = np.zeros((len(imgs), (n_coeffs * 4) - 1))

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
            # print(coeffs_to_keep)
            # print(fourier_coeffs.shape)
            # Calculate the elliptical Fourier descriptors.
            a0 = np.real(fourier_coeffs[0]) / n_samples
            b_coeffs = -np.imag(fourier_coeffs[1:]) / n_samples
            a_coeffs = np.real(fourier_coeffs[1:]) / n_samples
            # print(a0.shape, a_coeffs.shape, b_coeffs.shape)
            efds_list = [a0]
            efds_list.extend(np.ravel(a_coeffs))
            efds_list.extend(np.ravel(b_coeffs))
            # print(len(efds_list))
            efds[i] = np.array(efds_list)

        return efds
    
    def resample_contour(self, contour, n_samples):

        # Calculate the arc length of the contour.
        arc_length = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
        arc_length = np.insert(arc_length, 0, 0) / arc_length[-1]

        # Create a uniformly spaced grid of points along the arc length.
        t = np.linspace(0, 1, n_samples)

        # Interpolate the contour points along the arc length.
        x = np.interp(t, arc_length, contour[:, 0])
        y = np.interp(t, arc_length, contour[:, 1])

        return np.column_stack((x, y))

    def extract_efds_features(self, images):
        # Load an image and extract a contour
        coeffs = []
        for image in images:    
            binary_image = cv2.adaptiveThreshold(image, maxValue=255, 
                                                 adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                thresholdType=cv2.THRESH_BINARY_INV,
                                                    blockSize=11, C=2)
            contour,_ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contour = contour[0]
            print(contour)
            # print(np.all(binary_image == 0))
            # plt.imshow(binary_image)
            # plt.show()
            # Compute the EFDs for the contour
            num_coeff = 20
            x = contour[:, 0, 0]
            y = contour[:, 0, 1]
            coeff = elliptic_fourier_descriptors(np.column_stack((x, y)), order=num_coeff, normalize=True)
            coeffs.append(coeff.flatten())
        return np.array(coeffs)
        

