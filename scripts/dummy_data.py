import numpy as np
import matplotlib.pyplot as plt
import os

def create_dummy_data(num_samples, img_size=(256, 256, 3)):
    inputs = np.random.rand(num_samples, *img_size).astype(np.float32)  # Random images for input
    targets = np.random.rand(num_samples, *img_size).astype(np.float32)  # Random images for target
    return inputs, targets

num_samples = 6
input_images, target_images = create_dummy_data(num_samples)


#Ensure the data directory exists

os.makedirs('data', exist_ok=True)

plt.imsave(f'data/motion.jpg', input_images[0])
plt.imsave(f'data/skop1.jpg', target_images[0])
plt.imsave(f'data/Skop.jpg', target_images[1])
plt.imsave(f'data/skop2.jpg', target_images[2])
plt.imsave(f'data/skop3.jpg', target_images[3])
plt.imsave(f'data/skop4.jpg', target_images[4])
