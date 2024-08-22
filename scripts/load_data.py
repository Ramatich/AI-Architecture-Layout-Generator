import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob
from PIL import Image  # Don't forget to import Image from PIL

from train import get_script_dir

# Load images from the dataset folder
def load_dataset_data(dataset_dir, img_size=(256, 256)):
    image_paths = glob.glob(os.path.join(dataset_dir, '*.png'))
    images = []
    for path in image_paths:
        img = Image.open(path).resize(img_size)
        img = np.array(img).astype(np.float32) / 255.0  # Normalize images to [0, 1]
        images.append(img)
    return np.array(images), np.array(images)  # Using the same images as inputs and targets

# Load real data
data_dir = os.path.join(get_script_dir(), 'data', 'dataset')
input_images, target_images = load_dataset_data(data_dir)

