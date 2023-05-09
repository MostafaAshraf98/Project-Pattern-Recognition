from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

class DataLoader:
    
    def __init__(self, path: Path):
        self.path = path
        self.genders = ["men", "women"]
    
    def loadData(self):
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
                labels= []
                for img_path in digit_path.glob('*.JPG'):
                    try:
                        img = Image.open(img_path)
                        img = img.resize((224, 224))
                        images.append(np.array(img))
                        labels.append(digit)
                    except:
                        print("Image {} is corrupted".format(img_path))
                        continue
                    
                x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(images, labels, test_size=0.15, random_state=42)
                x_train_temp, x_val_temp, y_train_temp, y_val_temp = train_test_split(x_train_temp, y_train_temp, test_size=0.15/0.85, random_state=42)
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
        
        return x_train, y_train, x_test, y_test,x_val, y_val
    
dataloader = DataLoader(Path('./data'))
X_train, y_train, X_test, y_test, X_val, y_val = dataloader.loadData()
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")