"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import pathlib
import numpy as np
import tensorflow as tf
import keras

from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D


def main():

    # Create a tensorflow dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    val_data_dir = pathlib.Path('./data/tiny-imagenet-200/validation/')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    val_names = np.array([item.name for item in val_data_dir.glob('*')])
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

    val_data_gen = image_generator.flow_from_directory(directory=str(val_data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(val_names))

    # Create a simple model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Reshape((64*64*3,), input_shape=(64, 64, 3)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax'),
    # ])

    input = Input(shape=(64,64,3),name = 'image_input')

    vgg16_model = VGG16(weights='imagenet', include_top=False)
    out_vgg = vgg16_model(input)

    x = Flatten(name='flatten')(out_vgg)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(200, activation='sigmoid', name='predictions')(x)

    model = Model(input=input, output=x)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-6),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the simple model
    history = model.fit_generator(generator=train_data_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=1, validation_data=val_data_gen)
    # 28.66% accuracy with epoch = 10
    print(history.history.keys())
    print(history.history.values())

    # Save the final output
    model.save('model.h5')

if __name__ == '__main__':
    main()
