from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from PIL import ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from illumination_preprocessing.illumination_preprocessing import IlluminationPreprocessing
import tqdm

WIDTH = 320
HEIGHT = 320


class DataLoader:
    def __init__(self, path: Path):
        self.path = path
        self.genders = ["men", "Women"]
        self.desired_size = (WIDTH, HEIGHT)
        self.illumination_processing = IlluminationPreprocessing()

        
    def load_data(self, data_augmentation=False):
        try:
            return self.load_saved_data()
        except:
            return self.start_load_data(data_augmentation)


    def start_load_data(self, data_augmentation=False):
        images = []
        labels = []
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        x_val = []
        y_val = []

        for gender in self.genders:
            for digit in range(6):
                print(f"current digit: {digit}")
                digit_path = self.path / gender / str(digit)
                images = []
                labels = []
                for img_path in digit_path.glob("*.JPG"):
                    try:
                        img = Image.open(img_path)
                        images.append(np.array(img))
                        labels.append(digit)
                    except:
                        print("Image {} is corrupted".format(img_path))
                        continue

                images = np.array(images)
                
                images = self.resize_images(images)

                x_train_temp, x_val_temp, y_train_temp, y_val_temp = train_test_split(
                    images, labels, test_size=0.2, random_state=42
                )
                # x_train_temp, x_val_temp, y_train_temp, y_val_temp = train_test_split(
                #     x_train_temp, y_train_temp, test_size=0.1 / 0.9, random_state=42
                # )
                x_train.extend(x_train_temp)
                y_train.extend(y_train_temp)
                # x_test.extend(x_test_temp)
                x_val.extend(x_val_temp)
                y_val.extend(y_val_temp)
                # y_test.extend(y_test_temp)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        # x_test = np.array(x_test)
        # y_test = np.array(y_test)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        
        # shuffle the training data
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # shuffle the validation data
        indices = np.arange(x_val.shape[0])
        np.random.shuffle(indices)
        x_val = x_val[indices]
        y_val = y_val[indices]

        if (data_augmentation):
            x_train, y_train = self.data_augmentation(x_train, y_train)
            y_train = np.reshape(y_train, (y_train.shape[0], 1))
            
        self.save_data(x_train, y_train, x_test, y_test, x_val, y_val)

        return x_train, y_train, x_test, y_test, x_val, y_val

    def save_data(self, x_train, y_train, x_test, y_test, x_val, y_val):
        np.save("./data/x_train.npy", x_train)
        np.save("./data/y_train.npy", y_train)
        np.save("./data/x_test.npy", x_test)
        np.save("./data/y_test.npy", y_test)
        np.save("./data/x_val.npy", x_val)
        np.save("./data/y_val.npy", y_val)

    def load_saved_data(self):
        x_train = np.load("./data/x_train.npy")
        y_train = np.load("./data/y_train.npy")
        x_test = np.load("./data/x_test.npy")
        y_test = np.load("./data/y_test.npy")
        x_val = np.load("./data/x_val.npy")
        y_val = np.load("./data/y_val.npy")
    
        return x_train, y_train, x_test, y_test, x_val, y_val
    
    def data_augmentation(self, x_train, y_train):
        #--------------------------------------DATA AUGMENTATION-----------------------------------------

        # Define image data generator for data augmentation
        datagen = ImageDataGenerator(
            # rotation_range=40,  # randomly rotate images by up to 40 degrees
            width_shift_range=0.3,  # randomly shift images horizontally by up to 30%
            height_shift_range=0.3,  # randomly shift images vertically by up to 30%
            shear_range=0.2,  # randomly apply shearing transformations
            zoom_range=0.3,  # randomly zoom in on images by up to 30%
            channel_shift_range=20,  # randomly adjust brightness
            brightness_range=[0.5, 1.5],  # randomly adjust brightness
            horizontal_flip=True,  # randomly flip images horizontally
            vertical_flip=True,  # randomly flip images vertically
            fill_mode='nearest'  # fill in any gaps with the nearest pixel value
        )

        # Fit the data generator to your training data
        datagen.fit(x_train)

        # Define a function to generate augmented images and labels

        # Set batch size for training
        batch_size = 300

        # Generate augmented images and labels using the function defined above
        augmented_data = self.generate_augmented_data(x_train, y_train, batch_size, datagen)
        x_augmented, y_augmented = next(augmented_data)

        # Concatenate the original training set and the augmented images
        x_train = np.concatenate((x_train, x_augmented))

        # Concatenate the original labels and the augmented labels
        y_train = np.concatenate((y_train, y_augmented))
        y_train = np.reshape(y_train,(y_train.shape[0],1))

        return x_train, y_train

    
    
    def generate_augmented_data(self, x, y, batch_size, datagen):
        gen = datagen.flow(x, y, batch_size=batch_size)
        while True:
            x_batch, y_batch = gen.next()
            yield x_batch, y_batch
            
    def resize_images(self,images):
        images_resized = []
        for img in images:
            img = self.custom_resize_img(img)
            images_resized.append(img)
        
        images_resized = np.array(images_resized)
        return images_resized
            
            
            
    def custom_resize_img(self,img):
        # Calculate the aspect ratio of the image
        img = Image.fromarray(img)
        img_width, img_height   = img.size
        aspect_ratio = float(img_width) / float(img_height)

        # resize the image so that the shortest side is equal to the desired size
        if img_width < img_height:
            new_width = int(self.desired_size[0] * aspect_ratio)
            img = img.resize((new_width, self.desired_size[1]))
        else:
            new_height = int(self.desired_size[1] / aspect_ratio)
            img = img.resize((self.desired_size[0], new_height))

        # add padding to the image so that it is the desired size
        delta_width = self.desired_size[0] - img.size[0]
        delta_height = self.desired_size[1] - img.size[1]
        left = int(delta_width / 2)
        top = int(delta_height / 2)
        right = self.desired_size[0] - img.size[0] - left
        bottom = self.desired_size[1] - img.size[1] - top
        img = ImageOps.expand(
            img, border=(left, top, right, bottom), fill=(255,255,255)
        )
        img = np.array(img)
        return img