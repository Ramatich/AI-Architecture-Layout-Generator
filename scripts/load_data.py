import os
import numpy as np
from PIL import Image
import glob

# Function to get the directory of the current script
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

# Load images from the dataset folder
def load_dataset_data(input_dir, output_dir, img_size=(256, 256)):
    # Collect image paths for inputs and outputs
    input_image_paths = sorted(glob.glob(os.path.join(input_dir, '*.[pP][nN][gG]')) + glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]')))
    output_image_paths = sorted(glob.glob(os.path.join(output_dir, '*.[pP][nN][gG]')) + glob.glob(os.path.join(output_dir, '*.[jJ][pP][gG]')))

    input_dict = {}
    
    # Organize inputs by their base names (without extension)
    for img_path in input_image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if base_name not in input_dict:
            input_dict[base_name] = []
        img = Image.open(img_path).convert('RGB')
        img = img.resize(img_size)  # Resize to target size
        input_dict[base_name].append(np.array(img) / 127.5 - 1.0)

    inputs = []
    outputs = []

    # Match each output with its corresponding inputs
    for img_path in output_image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if base_name in input_dict:
            output_img = Image.open(img_path).convert('RGB')
            output_img = output_img.resize(img_size)  # Resize to target size
            output_array = np.array(output_img) / 127.5 - 1.0
            
            # Add each corresponding input and the output to the lists
            for input_array in input_dict[base_name]:
                inputs.append(input_array)
                outputs.append(output_array)

    return np.array(inputs), np.array(outputs)

# Paths to the input and output folders
data_dir = os.path.join(get_script_dir(), '..', 'data', 'dataset')
input_dir = os.path.join(data_dir, 'input')
output_dir = os.path.join(data_dir, 'output')

# Load data
input_images, target_images = load_dataset_data(input_dir, output_dir)

# Confirm data loading
print(f"Loaded {len(input_images)} images as input/target pairs from {input_dir} and {output_dir}")
