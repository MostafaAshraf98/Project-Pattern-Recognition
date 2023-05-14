from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import regularizers

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans

from skimage.filters import sobel, prewitt, roberts, laplace , median, gaussian, threshold_otsu, rank, threshold_local
from skimage.feature import canny
from skimage.restoration import denoise_nl_means, wiener
from skimage.morphology import disk , square
from skimage.draw import rectangle
from skimage import exposure, filters
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.morphology import closing, disk, skeletonize
from skimage.util import invert
from skimage.segmentation import clear_border
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.feature import daisy

from scipy.fftpack import fft
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from scipy import ndimage as ndi

from pathlib import Path
from PIL import Image, ImageOps
from pyefd import elliptic_fourier_descriptors
# from multiprocessing import Pool
from colorama import Fore, Back, Style
from skfuzzy.cluster import cmeans

import pickle
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
import hmmlearn.hmm as hmm
import datetime
import colorama



