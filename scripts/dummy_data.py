import numpy as np
import matplotlib.pyplot as plt

def create_dummy_data(num_samples, img_size=(256, 256, 3)):
    inputs = np.random.rand(num_samples, *img_size).astype(np.float32)  # Random images for input
    targets = np.random.rand(num_samples, *img_size).astype(np.float32)  # Random images for target
    return inputs, targets

num_samples = 100  # Number of dummy images
input_images, target_images = create_dummy_data(num_samples)

# Optionally, save a few dummy images to visualize
for i in range(5):
    plt.imsave(f'data/dummy_input_{i}.png', input_images[i])
    plt.imsave(f'data/dummy_target_{i}.png', target_images[i])
