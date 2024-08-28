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

def load_test_data(dataset_dir, img_size=(256, 256)):
    # Ensure dataset_dir exists and is a directory
    if not os.path.isdir(dataset_dir):
        print(f"Error: {dataset_dir} is not a directory or does not exist.")
        return np.array([])  # Return an empty array

    # Collect image paths
    image_paths = glob.glob(os.path.join(dataset_dir, '*.png')) + glob.glob(os.path.join(dataset_dir, '*.PNG'))
    print(f"Found images: {image_paths}")

    images = []
    
    if not image_paths:
        print("No images found in the dataset directory.")
        return np.array(images)  # Return an empty array if no images are found
    
    for path in image_paths:
        try:
            print(f"Loading image: {path}")
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(img_size)
            img = np.array(img).astype(np.float32) / 127.5 - 1.0  # Normalize images to [-1, 1]
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    return np.array(images)

# Example usage
print(f"Using dataset directory: {dataset_dir}")
images = load_test_data(dataset_dir)

if images.size > 0:
    print(f"Number of images loaded: {len(images)}")
else:
    print("No images were loaded.")

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

    # Load the trained generator model
    model_path = os.path.join(script_dir, 'pix2pix_model.h5')
    generator = load_model(model_path)
    if generator is None:
        print("Failed to load the model. Exiting.")
        return

    # Load test images
    test_dir = '/content/AI-Architecture-Layout-Generator/data/dataset'

    test_images = load_test_data(test_dir)
    
    if test_images.size == 0:
        print(f"No images to process in {test_dir}. Exiting.")
        return

    print(f"Loaded {len(test_images)} images from {test_dir}")

    # Generate images
    generated_images = generate_images(generator, test_images)
    
    if generated_images.size == 0:
        print("No images generated. Exiting.")
        return

    # Save the original and generated images
    output_dir = os.path.join(script_dir, 'generated_images')
    save_images(test_images, generated_images, output_dir)
    print(f"Images saved in {output_dir}")

if __name__ == "__main__":
    main()
