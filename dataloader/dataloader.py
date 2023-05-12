from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from PIL import ImageOps

WIDTH = 640
HEIGHT = 640


class DataLoader:
    def __init__(self, path: Path):
        self.path = path
        self.genders = ["men", "women"]
        self.desired_size = (WIDTH, HEIGHT)
        
    def load_data(self):
        try:
            return self.load_saved_data()
        except:
            return self.start_load_data()


    def start_load_data(self):
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
                digit_path = self.path / gender / str(digit)
                images = []
                labels = []
                for img_path in digit_path.glob("*.JPG"):
                    try:
                        img = Image.open(img_path)

                        # Calculate the aspect ratio of the image
                        img_width, img_height = img.size
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

                        images.append(np.array(img))
                        labels.append(digit)
                    except:
                        print("Image {} is corrupted".format(img_path))
                        continue

                x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(
                    images, labels, test_size=0.15, random_state=42
                )
                x_train_temp, x_val_temp, y_train_temp, y_val_temp = train_test_split(
                    x_train_temp, y_train_temp, test_size=0.15 / 0.85, random_state=42
                )
                x_train.extend(x_train_temp)
                y_train.extend(y_train_temp)
                x_test.extend(x_test_temp)
                y_test.extend(y_test_temp)
                x_val.extend(x_val_temp)
                y_val.extend(y_val_temp)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

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
    
