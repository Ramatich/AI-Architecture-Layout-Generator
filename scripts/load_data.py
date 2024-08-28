import os
import numpy as np
from PIL import Image
import glob

# Function to get the directory of the current script
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

# Load images from the dataset folder
def load_dataset_data(dataset_dir, img_size=(256, 256)):
    # Collect image paths (including different formats if needed)
    image_paths = glob.glob(os.path.join(dataset_dir, '*.[pP][nN][gG]')) + glob.glob(os.path.join(dataset_dir, '*.[jJ][pP][gG]'))
    images = []
    
    for path in image_paths:
        img = Image.open(path)
        
        # Convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(img_size)
        img = np.array(img).astype(np.float32) / 127.5 - 1.0  # Normalize images to [-1, 1]
        images.append(img)
    
    return np.array(images), np.array(images)  # Using the same images as inputs and targets

# Load real data
data_dir = os.path.join(get_script_dir(), '..', 'data', 'dataset')
input_images, target_images = load_dataset_data(data_dir)

# Confirm data loading
print(f"Loaded {len(input_images)} images as input/target pairs from {data_dir}")
