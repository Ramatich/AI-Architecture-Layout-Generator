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
    if not image_paths:
        print("No images found in the dataset directory.")
        return np.array(images)  # Return an empty array if no images are found
    
    for path in image_paths:
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(img_size)
            img = np.array(img).astype(np.float32) / 255.0  # Normalize images to [0, 1]
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    
    return np.array(images)

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
            # Save original image
            original_image_path = os.path.join(output_dir, f"original_image_{i+1}.png")
            plt.imsave(original_image_path, np.clip(test_images[i], 0, 1))

            # Save generated image
            generated_image_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
            plt.imsave(generated_image_path, np.clip(generated_images[i], 0, 1))
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
    test_dir = os.path.join(script_dir, '..', 'data', 'dataset')
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
