from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.color import label2rgb
from sklearn.cluster import KMeans
from skimage.morphology import closing, disk, skeletonize
from skimage.util import invert
from skimage.segmentation import clear_border
from skfuzzy.cluster import cmeans
import numpy as np


class ClusteringSegmentation:
        
    def __init__(self, method='kmeans', n_clusters=3, compactness=30.0, sigma=1.0):

        # method: 'kmeans' or 'fcm'
        
        self.method = method
        self.n_clusters = n_clusters
        self.compactness = compactness
        self.sigma = sigma

    def process(self, image):
        # convert the image to grayscale

        # apply SLIC algorithm to get superpixels
        scaled_image = img_as_float(resize(image, (500, 500)))
        segments = slic(scaled_image, n_segments=300, compactness=self.compactness, sigma=self.sigma)

        
        # calculate the color features of each superpixel
        features = []
        for i in np.unique(segments):
            mask = segments == i    
            feature = np.mean(scaled_image[mask])
            features.append(feature)
        features = np.array(features)

        # cluster the superpixels based on their color features
        if self.method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters).fit(features.reshape(-1, 1))
            labels = kmeans.labels_

        elif self.method == 'fcm':
            gray_image = rgb2gray(image)

            # Apply a threshold to the image
            thresh = threshold_otsu(gray_image)
            binary = gray_image > thresh

            # Apply morphological operations to clean up the image
            binary = closing(binary, disk(2))
            binary = clear_border(binary)

            # Obtain a skeleton of the binary image
            skeleton = skeletonize(binary)

            # Apply fuzzy logic to the skeleton
            fuzzy_skeleton = gaussian(skeleton, sigma=2)
            data = fuzzy_skeleton.reshape(-1, 1)
            cntr, u, u0, d, jm, p, fpc = cmeans(data.T, 2, 2, error=0.005, maxiter=1000)
            fuzzy_labels = np.argmax(u, axis=0).reshape(image.shape[:2])
            return fuzzy_labels
            

        # create an image with each superpixel labeled by its cluster
        labels = np.array(labels, dtype=np.float64)
        label_image = np.zeros_like(segments)
        for i, label in enumerate(np.unique(segments)):
            mask = segments == label
            label_image[mask] = labels[i]

        # post-process the labeled image
        label_image = label_image.astype(int)
        regions = regionprops(label_image)
        for i, region in enumerate(regions):
            if region.area < 500:
                label_image[label_image == i+1] = 0

        # color the regions for visualization
        # labeled_regions = label2rgb(label_image, image=image, bg_label=0, kind='avg')
        labeled_regions = label2rgb(label_image, image=scaled_image, bg_label=0, kind='avg')

        return labeled_regions
