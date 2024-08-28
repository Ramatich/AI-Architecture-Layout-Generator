import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Function to get the directory of the current script
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.join('/content/AI-Architecture-Layout-Generator', 'data', 'dataset')
print(f"Looking for images in: {os.path.abspath(dataset_dir)}")

def load_test_data(input_dir, output_dir, img_size=(256, 256)):
    # Ensure input_dir and output_dir exist and are directories
    if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
        print(f"Error: One or both directories do not exist.")
        return np.array([]), np.array([])  # Return empty arrays

    # Collect image paths
    input_image_paths = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.PNG'))
    output_image_paths = glob.glob(os.path.join(output_dir, '*.png')) + glob.glob(os.path.join(output_dir, '*.PNG'))
    
    print(f"Found input images: {input_image_paths}")
    print(f"Found output images: {output_image_paths}")

    if not input_image_paths or not output_image_paths:
        print("No images found in one or both directories.")
        return np.array([]), np.array([])

    inputs = []
    outputs = []

    for path in input_image_paths:
        try:
            print(f"Loading input image: {path}")
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(img_size)
            img = np.array(img).astype(np.float32) / 127.5 - 1.0  # Normalize images to [-1, 1]
            inputs.append(img)
        except Exception as e:
            print(f"Error loading input image {path}: {e}")

    for path in output_image_paths:
        try:
            print(f"Loading output image: {path}")
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(img_size)
            img = np.array(img).astype(np.float32) / 127.5 - 1.0  # Normalize images to [-1, 1]
            outputs.append(img)
        except Exception as e:
            print(f"Error loading output image {path}: {e}")

    return np.array(inputs), np.array(outputs)

# Example usage
print(f"Using dataset directory: {dataset_dir}")

input_dir = os.path.join(dataset_dir, 'input')
output_dir = os.path.join(dataset_dir, 'output')

test_images, target_images = load_test_data(input_dir, output_dir)

if test_images.size > 0:
    print(f"Number of input images loaded: {len(test_images)}")
else:
    print("No input images were loaded.")

if target_images.size > 0:
    print(f"Number of output images loaded: {len(target_images)}")
else:
    print("No output images were loaded.")

# Load the trained model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# Generate images using the generator model
def generate_images(generator, test_images):
    try:
        return generator.predict(test_images)
    except Exception as e:
        print(f"Error during image generation: {e}")
        return np.array([])

# Save the original and generated images
def save_images(test_images, generated_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_images = min(len(test_images), len(generated_images))

    for i in range(num_images):
        try:
            # Save original image (re-normalize to [0, 1])
            original_image_path = os.path.join(output_dir, f"original_image_{i+1}.png")
            plt.imsave(original_image_path, (test_images[i] + 1.0) / 2.0)

            # Save generated image (re-normalize to [0, 1])
            generated_image_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
            plt.imsave(generated_image_path, (generated_images[i] + 1.0) / 2.0)
        except Exception as e:
            print(f"Error saving image {i+1}: {e}")

def main():
    # Get the directory of the current script
    script_dir = get_script_dir()

    # Set input and output directories
    input_dir = os.path.join(script_dir, 'data', 'dataset', 'input')
    output_dir = os.path.join(script_dir, 'data', 'dataset', 'output')  # Initialize output_dir here
    
    # Load the trained generator model
    model_path = os.path.join(script_dir, 'pix2pix_model.h5')
    generator = load_model(model_path)
    if generator is None:
        print("Failed to load the model. Exiting.")
        return

    # Load test images
    test_images, target_images = load_test_data(input_dir, output_dir)
    
    if test_images.size == 0:
        print(f"No images to process in {input_dir}. Exiting.")
        return

    print(f"Loaded {len(test_images)} input images from {input_dir}")

    # Generate images
    generated_images = generate_images(generator, test_images)
    
    if generated_images.size == 0:
        print("No images generated. Exiting.")
        return

    # Save the original and generated images
    output_dir = os.path.join(script_dir, 'generated_images')  # Reassign output_dir for saving images
    save_images(test_images, generated_images, output_dir)
    print(f"Images saved in {output_dir}")

