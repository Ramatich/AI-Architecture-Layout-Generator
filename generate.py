import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_generator

# Load the trained generator model
generator = build_generator()
generator.load_weights('pix2pix_model_generator.h5')

def generate_layout(input_image):
    # Ensure input_image is preprocessed to shape (256, 256, 3)
    input_image = (input_image * 2) - 1  # Normalize input image to [-1, 1]
    generated_layout = generator.predict(np.expand_dims(input_image, axis=0))
    return generated_layout[0]

# Generate and visualize a layout
input_image = np.random.rand(256, 256, 3).astype(np.float32)  # Dummy input image
generated_layout = generate_layout(input_image)

# Save and display the generated layout
plt.imsave('generated_layout.png', (generated_layout + 1) / 2)  # Denormalize to [0, 1]
plt.imshow((generated_layout + 1) / 2)
plt.show()
