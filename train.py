import os
import numpy as np
import tensorflow as tf
from model import build_generator, build_discriminator, compile_pix2pix
from scripts.load_data import load_dataset_data
import matplotlib.pyplot as plt

# Function to get the directory of the current script
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

# Load real data from the dataset
data_dir = os.path.join(get_script_dir(), 'data', 'dataset')
input_images, target_images = load_dataset_data(data_dir)

# Create models
generator = build_generator()
discriminator = build_discriminator()
generator, discriminator = compile_pix2pix(generator, discriminator)

# Training parameters
epochs = 2  # You can adjust the number of epochs as needed
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