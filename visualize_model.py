import os
from model import build_generator, build_discriminator
from tensorflow.keras.utils import plot_model

# Function to get the directory of the current script
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

# Create models
generator = build_generator()
discriminator = build_discriminator()

# Get the directory of the current script
script_dir = get_script_dir()

# Define paths for saving the model visualizations
generator_plot_path = os.path.join(script_dir, 'generator_plot.png')
discriminator_plot_path = os.path.join(script_dir, 'discriminator_plot.png')

# Plot the models
plot_model(generator, to_file=generator_plot_path, show_shapes=True, show_layer_names=True)
plot_model(discriminator, to_file=discriminator_plot_path, show_shapes=True, show_layer_names=True)

print(f"Generator model visualization saved at {generator_plot_path}")
print(f"Discriminator model visualization saved at {discriminator_plot_path}")
