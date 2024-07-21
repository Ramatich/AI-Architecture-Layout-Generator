import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load dummy data
def load_dummy_data():
    num_samples = 5  # Match this with the number of saved images
    img_size = (256, 256, 3)
    inputs = np.zeros((num_samples, *img_size), dtype=np.float32)
    targets = np.zeros((num_samples, *img_size), dtype=np.float32)

    for i in range(num_samples):
        input_img = Image.open(f'data/input_image_{i}.jpg').resize(img_size[:2])
        target_img = Image.open(f'data/target_image_{i}.jpg').resize(img_size[:2])
        inputs[i] = np.array(input_img) / 255.0  # Normalize the images
        targets[i] = np.array(target_img) / 255.0  # Normalize the images

    return inputs, targets

input_images, target_images = load_dummy_data()

# Load the trained model
pix2pix_model = load_model('pix2pix_model.h5')

# Test the model with dummy data
predictions = pix2pix_model.predict([input_images, input_images])  # Here, using inputs for both input and target images

# Display results
for i in range(len(predictions)):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(input_images[i])
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Target")
    plt.imshow(target_images[i])
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(predictions[i])
    plt.axis('off')

    plt.show()
