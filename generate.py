import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_generator
from PIL import Image
import os

# Load the trained generator model
generator = build_generator()
generator.load_weights('pix2pix_model.h5')

def preprocess_image(image_path):
    """Load and preprocess an image from the given path."""
    image = Image.open(image_path).resize((256, 256))
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image

def generate_layout(input_image):
    """Generate a layout using the trained model."""
    input_image = (input_image * 2) - 1  # Normalize input image to [-1, 1]
    generated_layout = generator.predict(np.expand_dims(input_image, axis=0))
    return generated_layout[0]

def save_and_display_image(image, filename):
    """Save and display the generated image."""
    plt.imsave(filename, (image + 1) / 2)  # Denormalize to [0, 1]
    plt.imshow((image + 1) / 2)
    plt.axis('off')
    plt.show()

def process_images(input_folder, output_folder):
    """Process images from the input folder and save the generated layouts to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each image in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_image = preprocess_image(image_path)
            generated_layout = generate_layout(input_image)
            output_filename = os.path.join(output_folder, image_name.replace('.png', '_generated_layout.png'))
            save_and_display_image(generated_layout, output_filename)

# Paths to input and output folders
data_folder = 'data'
dataset_folder = os.path.join(data_folder, 'dataset')
input_folder = os.path.join(dataset_folder, 'test_input')
output_folder = os.path.join(dataset_folder, 'generated_layouts')

# Process images
process_images(input_folder, output_folder)
