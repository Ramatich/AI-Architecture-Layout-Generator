from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob

def load_real_data(data_dir, img_size=(256, 256)):
    image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))  # Adjust the extension if necessary
    images = []
    for path in image_paths:
        img = load_img(path, target_size=img_size)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images)

# Load real data
data_dir = os.path.join(get_script_dir(), 'data', 'dataset')
input_images = load_real_data(data_dir)
target_images = load_real_data(data_dir)  # Replace this with the appropriate target data loading
