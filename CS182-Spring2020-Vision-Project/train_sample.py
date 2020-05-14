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

from keras.utils.vis_utils import plot_model

from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge

from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout
from utils.log import log, TimeHistory, cparams

def main():
    # Create a tensorflow dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200/b_train/')
    val_data_dir = pathlib.Path('./data/tiny-imagenet-200/validation/')

    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    val_names = np.array([item.name for item in val_data_dir.glob('*')])

    print('Discovered {} images'.format(image_count))

    # Simple image preprocessing:  Scale images to [0, 1] float32
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)


    # Create the training data generator
    BATCH_SIZE = 32
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)
    train_data_gen = image_gen.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(CLASS_NAMES),
                                                         interpolation='bicubic',
                                                         seed=7)


    val_data_gen = image_gen.flow_from_directory(directory=str(val_data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         interpolation='bicubic',
                                                         classes=list(val_names))


    input = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3), name = 'image_input')

    image_model = InceptionV3(weights='imagenet', include_top=False)
    image_model.summary()
    image_model = image_model(input)


    x = Flatten(name='flatten')(image_model)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(200, activation='sigmoid', name='predictions')(x)

    model = Model(input=input, output=x)
    plot_model(model, to_file='incep3.png', show_shapes=True)

    model.summary()

    lr = 1e-6
    epochs = 15
    model_name = 'incep3'



    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    # Train the simple model
    time_callback = TimeHistory()
    history = model.fit_generator(generator=train_data_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=epochs, validation_data=val_data_gen, callbacks=[time_callback])

    # Save the final output
    model.save('./models/' + model_name + '.h5')


    # For logging
    logged_params = {'name': model_name, 'lr': lr, 'epochs': epochs, 'time' : round(sum(time_callback.times))}
    log(history.history, logged_params)


if __name__ == '__main__':
    main()
