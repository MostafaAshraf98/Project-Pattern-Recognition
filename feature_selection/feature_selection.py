import sys
sys.path.append('../')

from imports import *


class FeatureSelector:
    def __init__(self) -> None:
        pass

    def extract_pca_features(self, images, load=False, num_pca_components=0.95):
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
        
        if load:
            pca = pickle.load(open("pca.pkl", "rb"))
            pca_features = pca.transform(image_vectors)
            return pca_features
        else:
            print("Creating new PCA model...")
            pca = PCA(n_components = num_pca_components, svd_solver = 'full')
            pca.fit(image_vectors)

            pca_features = pca.transform(image_vectors)

            pca_features = np.array(pca_features)
            
            pickle.dump(pca, open("pca.pkl", "wb"))
            
            return pca_features
        
    def test_pca(self,img, pca):
        image_vector = img.flatten()
        pca_features = pca.transform(np.array([image_vector]))
        return pca_features
        
