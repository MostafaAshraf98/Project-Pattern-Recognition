import sys
sys.path.append('../')

from imports import *

class ImageAligner:
    def init(self):
        pass

    def align_image(self, binary_images):
        
        aligned_images = []
        for image in binary_images:
            # Apply Canny edge detection to the image
            edges = cv2.Canny(image, 20, 100)
            # plt.imshow(edges, cmap='gray')
            # plt.show()

            # print(np.all(edges == 0))
            # print('Edges:', edges)
            # Detect the lines in the image using the Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
            
            # Count the number of lines with each angle
            angle_counts = {}
            # print('Lines', lines)
            if (not (lines is None)):
                for line in lines:
                    # print('Line', line)
                    rho, theta = line[0]
                    angle_degrees = int(theta * 180/np.pi)
                    # print('Angle degrees', angle_degrees)
                    if angle_degrees in angle_counts:
                        angle_counts[angle_degrees] += 1
                    else:
                        angle_counts[angle_degrees] = 1
            

                    # print(angle_counts)
                    # Find the angle with the highest count
                    most_frequent_angle = max(angle_counts, key=angle_counts.get)
                    # print("Most frequent angle:", most_frequent_angle)
                    # Rotate the image to align with the median angle
                    rows, cols = image.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), most_frequent_angle, 1)
                    aligned_image = cv2.warpAffine(image, M, (cols, rows))


                pixels_up = np.sum(np.sum(aligned_image, axis = 1)[:70])
                pixels_down = np.sum(np.sum(aligned_image, axis = 1)[-70:])
                # plt.imshow(aligned_image, cmap='gray')
                # plt.show()
                # print(pixels_up, pixels_down)
                if (pixels_up > pixels_down):
                    aligned_image = np.rot90(aligned_image, 2)            
            
                aligned_images.append(aligned_image)
            else:
                aligned_images.append(image)
    
        return np.array(aligned_images)
        