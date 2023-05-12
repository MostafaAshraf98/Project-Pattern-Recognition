from multiprocessing import Pool
from scipy import ndimage as ndi
import numpy as np
import cv2

class IlluminationPreprocessing:
  def __init__(self):
    pass
  
  def retinex(self, image, sigma=100):
      image = np.copy(image)
      # Convert image to float32
      image = image.astype(np.float32)

      # Compute logarithmic luminance
      log_luminance = np.log(image.mean(axis=2))

      # Compute multiscale decomposition using a Gaussian pyramid
      pyramid = []
      pyramid.append(log_luminance)
      for i in range(1, 3):
          pyramid.append(cv2.pyrDown(pyramid[i - 1]))

      # Compute local contrast for each scale
      contrast = []
      for i in range(3):
          laplacian = cv2.Laplacian(pyramid[i], cv2.CV_32F, ksize=3)
          contrast.append(np.exp(np.abs(cv2.resize(laplacian, log_luminance.shape[::-1])) / sigma))

      # Compute reflectance by multiplying local contrast across scales
      reflectance = np.ones_like(log_luminance)
      for i in range(3):
          reflectance *= cv2.resize(contrast[i], log_luminance.shape[::-1])

      # Compute illumination by dividing logarithmic luminance by reflectance
      illumination = np.exp(log_luminance) / reflectance

      # Rescale illumination to have the same range as the input image
      illumination = cv2.normalize(illumination, None, 0, 255, cv2.NORM_MINMAX)

      # Convert illumination back to uint8 and merge with original image
      illumination = illumination.astype(np.uint8)
      result = cv2.merge([image[:,:,0] - illumination, image[:,:,1] - illumination, image[:,:,2] - illumination])

      # Invert pixel values to range of [0, 255]
      result = (255 - result).clip(0, 255).astype(np.uint8)
      return result


  def process_image(self, img):
      op = self.retinex(img)
      op = cv2.cvtColor(op, cv2.COLOR_RGB2HSV)
      saturation_channel = op[:, :, 1]

      # Calculate actual threshold value
      _, actual_threshold = cv2.threshold(saturation_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      # plt.imshow(saturation_channel, cmap="gray")
      # plt.show()
      kernel = np.ones((5, 5), np.uint8)
      # Perform morphological operations to remove small noise and fill gaps in the binary image
      mask = cv2.morphologyEx(actual_threshold, cv2.MORPH_OPEN, kernel)
      mask = ndi.binary_fill_holes(mask).astype(np.uint8)
      # plt.imshow(mask, cmap="gray")
      # plt.show()
      # Apply binary mask to grayscale image
      saturation_channel_masked = cv2.multiply(saturation_channel, mask)
      # plt.imshow(saturation_channel_masked, cmap="gray")
      # plt.show()
      return saturation_channel_masked


  def process_images_multiprocess(self, images):
      with Pool() as p:
          results = p.map(self.process_image, images)
      return np.stack(results, axis=0)

  def process_images_loops(self, images):
      results = []
      for image in images:
          results.append(self.process_image(image))
      return results
