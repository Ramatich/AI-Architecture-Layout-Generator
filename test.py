import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import subprocess

# Function to run dummy_data.py
def generate_dummy_data():
    subprocess.run(['python', 'scripts/dummy_data.py'], check=True)

# Function to load and preprocess images
def load_images(image_paths, img_size=(256, 256)):
    images = []
    for path in image_paths:
        img = Image.open(path).resize(img_size)
        img = np.array(img).astype(np.float32) / 255.0  # Normalize images to [0, 1]
        images.append(img)
    return np.array(images)

# Function to display images
def display_images(images, titles):
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

# Generate dummy data
generate_dummy_data()

# Load and preprocess dummy images
image_paths = [
    'data/motion.jpg',
    'data/Skop.jpg',
    'data/skop2.jpg',
    'data/skop3.jpg',
    'data/skop4.jpg'
]
images = load_images(image_paths)

# Load the trained Pix2Pix model
model = tf.keras.models.load_model('pix2pix_model.h5', custom_objects={'pix2pix_loss': None})

# Generate predictions
predictions = model.predict(images)

# Display the results
display_images(images, ['Input'] * len(images))
display_images(predictions, ['Predicted'] * len(predictions))
