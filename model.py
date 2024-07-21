import tensorflow as tf
from tensorflow.keras import layers

def build_generator():
    inputs = layers.Input(shape=[256, 256, 3])
    
    # Encoder
    down1 = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu')(inputs)
    down2 = layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu')(down1)
    
    # Bottleneck
    bottleneck = layers.Conv2D(256, (4, 4), strides=2, padding='same', activation='relu')(down2)
    
    # Decoder
    up1 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu')(bottleneck)
    concat1 = layers.Concatenate()([up1, down2])
    up2 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(concat1)
    
    # Output layer
    outputs = layers.Conv2D(3, (7, 7), activation='tanh', padding='same')(up2)
    
    return tf.keras.Model(inputs, outputs)

def build_discriminator():
    inputs = layers.Input(shape=[256, 256, 3])
    target = layers.Input(shape=[256, 256, 3])
    
    x = layers.Concatenate()([inputs, target])
    
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu')(x)
    
    x = layers.Conv2D(256, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (4, 4), strides=1, padding='same', activation='relu')(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model([inputs, target], x)

def compile_pix2pix(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    
    inputs = layers.Input(shape=[256, 256, 3])
    target = layers.Input(shape=[256, 256, 3])
    generated = generator(inputs, training=True)
    
    disc_output = discriminator([inputs, generated], training=True)
    
    pix2pix_model = tf.keras.Model(inputs=[inputs, target], outputs=[disc_output, generated])
    pix2pix_model.compile(optimizer='adam', loss=['binary_crossentropy', 'mae'])
    
    return pix2pix_model
