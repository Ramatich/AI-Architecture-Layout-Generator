import os
import shutil
import re

def organize_files():
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(base_dir, 'data', 'dataset')
    input_dir = os.path.join(dataset_dir, 'input')
    output_dir = os.path.join(dataset_dir, 'output')

    # Create new directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Compile regex patterns
    input_pattern = re.compile(r'^\d+(?:_)?input\.png$')
    output_pattern = re.compile(r'^\d+output\.png$')

    # Iterate through files in the dataset directory
    for filename in os.listdir(dataset_dir):
        if input_pattern.match(filename):
            shutil.move(os.path.join(dataset_dir, filename), os.path.join(input_dir, filename))
            print(f"Moved {filename} to input directory")
        elif output_pattern.match(filename):
            shutil.move(os.path.join(dataset_dir, filename), os.path.join(output_dir, filename))
            print(f"Moved {filename} to output directory")

if __name__ == "__main__":
    organize_files()
    print("File organization complete.")