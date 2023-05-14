# Required Files in the same directory:
# 1. model.h5
# 2. pca.pkl
# 3. extracted_features_train_mean.npy
# 4. extracted_features_train_std.npy

from dataloader.dataloader import DataLoader
from feature_extraction.feature_extraction import FeatureExtractor
from feature_selection.feature_selection import FeatureSelector
from model_selection.model_selection import ModelSelection
from performance_analysis.performance_analysis import PerformanceAnalysis
from illumination_preprocessing.illumination_preprocessing import IlluminationPreprocessing
from preprocessing.image_aligner import ImageAligner

from imports import *

data_loader = DataLoader(Path('./test'))
illumination_processing = IlluminationPreprocessing()
feature_extractor = FeatureExtractor()
feature_selector = FeatureSelector()
image_aligner = ImageAligner()

model = load_model("model.h5")
pca = pickle.load(open("pca.pkl", "rb"))
extracted_features_train_mean = np.load("extracted_features_train_mean.npy")
extracted_features_train_std = np.load("extracted_features_train_std.npy")

path = Path('./test')

if os.path.exists("results.txt"):
    os.remove("results.txt")
if os.path.exists("time.txt"):
    os.remove("time.txt")

results_file = open("results.txt", "w")
time_file = open("time.txt", "w")

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Sort the list of files in increasing order
files.sort(key=lambda x: int(os.path.splitext(x)[0]))

# Loop over all the image files, read each image using cv2.imread and store it in the numpy array
for i, filename in enumerate(files):
    img = cv2.imread(os.path.join(path, filename))
    img = np.array(img)
    
    # Get current time
    start = time.perf_counter()
    
    # Resize the image
    img = data_loader.custom_resize_img(img)
    
    # Illumination Preprocessing
    illuminated_test, _ = illumination_processing.process_image(img)

    # Image Alignment
    aligned_test = image_aligner.align_image([illuminated_test])[0]

    # Feature extraction and selection
    daisy_features_test = feature_extractor.extract_daisy_features([aligned_test])[0]

    pca_daisy_features_test = feature_selector.test_pca(daisy_features_test,pca)
    
    pca_daisy_features_test = (pca_daisy_features_test - extracted_features_train_mean) /extracted_features_train_std

    # Model loading and prediction
    model_prediction = model.predict(pca_daisy_features_test)

    # Only in case of ANN
    model_prediction = model_prediction.argmax(axis=1)
    
    # stop timer
    end = time.perf_counter()
    
    total_time_seconds = round(end - start, 3)

    # write the prediction in results file
    results_file.write(f"{int(model_prediction[0])}\n")
    
    # write the time in times file
    time_file.write(f"{total_time_seconds}\n")

results_file.close()
time_file.close()