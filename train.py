import os
import numpy as np
import tensorflow as tf
from model import build_generator, build_discriminator, compile_pix2pix
from PIL import Image
from glob import glob

# Function to get the directory of the current script
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

# Load input and output images from the dataset
def load_dataset_data(input_dir, output_dir, target_size=(256, 256)):
    # List all input and output images
    input_images = sorted(glob(os.path.join(input_dir, '*input.png')))
    output_images = sorted(glob(os.path.join(output_dir, '*output.png')))

    print(f"Input images found: {input_images}")
    print(f"Output images found: {output_images}")

    input_dict = {}

    # Organize inputs by their base names (without '_input' suffix)
    for img_path in input_images:
        base_name = os.path.basename(img_path).replace('_input.png', '')
        input_dict[base_name] = img_path

    inputs, outputs = [], []

    # Match each output with its corresponding inputs
    for img_path in output_images:
        base_name = os.path.basename(img_path).replace('output.png', '')
        if base_name in input_dict:
            try:
                input_img = Image.open(input_dict[base_name]).convert('RGB')
                input_img = input_img.resize(target_size)
                output_img = Image.open(img_path).convert('RGB')
                output_img = output_img.resize(target_size)

                inputs.append(np.array(input_img) / 127.5 - 1.0)
                outputs.append(np.array(output_img) / 127.5 - 1.0)
            except Exception as e:
                print(f"Error processing {base_name}: {e}")

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    print(f"Loaded {len(inputs)} input images and {len(outputs)} output images.")

    return inputs, outputs



# Paths to the input and output folders
data_dir = os.path.join(get_script_dir(), 'data', 'dataset')
input_dir = os.path.join(data_dir, 'input')
output_dir = os.path.join(data_dir, 'output')

# Load data
input_images, target_images = load_dataset_data(input_dir, output_dir)

# Create models
generator = build_generator()
discriminator = build_discriminator()
generator, discriminator = compile_pix2pix(generator, discriminator)

# Training parameters
epochs = 1  # You can adjust the number of epochs as needed
batch_size = 1  # You may need to adjust this based on memory constraints

# Train the Pix2Pix model
for epoch in range(epochs):
    print(f'Starting epoch {epoch+1}')
    num_batches = len(input_images) // batch_size
    for i in range(num_batches):
        batch_input_images = input_images[i*batch_size:(i+1)*batch_size]
        batch_target_images = target_images[i*batch_size:(i+1)*batch_size]
        
        for input_image, target_image in zip(batch_input_images, batch_target_images):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_image = generator(tf.expand_dims(input_image, 0), training=True)
                real_output = discriminator([tf.expand_dims(input_image, 0), tf.expand_dims(target_image, 0)], training=True)
                fake_output = discriminator([tf.expand_dims(input_image, 0), generated_image], training=True)

                gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)
                disc_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
                disc_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
                disc_loss = disc_loss_real + disc_loss_fake

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator.optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print(f'Finished epoch {epoch+1}')

# Save the trained model in the same directory as train.py
script_dir = get_script_dir()
model_save_path = os.path.join(script_dir, 'pix2pix_model.h5')
generator.save(model_save_path)
print(f"Model saved at {model_save_path}")
