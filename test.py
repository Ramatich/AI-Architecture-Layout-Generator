import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Function to get the directory of the current script
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

# Load images from the test dataset folder
def load_test_data(dataset_dir, img_size=(256, 256)):
    image_paths = glob.glob(os.path.join(dataset_dir, '*.png'))
    images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(img_size)
        img = np.array(img).astype(np.float32) / 255.0  # Normalize images to [0, 1]
        images.append(img)
    
    return np.array(images)

# Load the trained model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Generate images using the generator model
def generate_images(generator, test_images):
    return generator.predict(test_images)

# Visualize original and generated images
def plot_images(test_images, generated_images, num_images=5):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 2, 2*i+1)
        plt.imshow(test_images[i])
        plt.title("Original Image")
        plt.axis('off')

        # Generated image
        plt.subplot(num_images, 2, 2*i+2)
        plt.imshow(generated_images[i])
        plt.title("Generated Image")
        plt.axis('off')
    
    plt.show()

def main():
    # Get the directory of the current script
    script_dir = get_script_dir()

    # Load the trained generator model
    model_path = os.path.join(script_dir, 'pix2pix_model.h5')
    generator = load_model(model_path)

    # Load test images
    test_dir = os.path.join(script_dir, '..', 'data', 'dataset')  # Update this path to your test dataset
    test_images = load_test_data(test_dir)

    # Generate images
    generated_images = generate_images(generator, test_images)

    # Visualize the results
    plot_images(test_images, generated_images)

if __name__ == "__main__":
    main()
