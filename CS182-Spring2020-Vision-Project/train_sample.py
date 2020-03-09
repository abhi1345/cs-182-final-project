"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import pathlib
import numpy as np
import tensorflow as tf


def main():

    # Create a tensorflow dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    print('Discovered {} images'.format(image_count))

    # Simple image preprocessing:  Scale images to [0, 1] float32
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # Create the training data generator
    BATCH_SIZE = 32
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(CLASS_NAMES))


    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((64*64*3,), input_shape=(64, 64, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(CLASS_NAMES)),
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the simple model
    model.fit_generator(generator=train_data_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=1)

    # Save the final output
    model.save('model.h5')


if __name__ == '__main__':
    main()