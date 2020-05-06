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
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D
from utils.log import log, TimeHistory, cparams
# from fool_image import fool_image

def main():
    # Create a tensorflow dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200/new_train/')
    val_data_dir = pathlib.Path('./data/tiny-imagenet-200/validation/')

    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    val_names = np.array([item.name for item in val_data_dir.glob('*')])

    print('Discovered {} images'.format(image_count))

    # Simple image preprocessing:  Scale images to [0, 1] float32
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    train_gen2 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)


    # Create the training data generator
    BATCH_SIZE = 32
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)
    train_data_gen = train_gen.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(CLASS_NAMES),
                                                         seed=7)

    train_data_gen2 = train_gen2.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(CLASS_NAMES),
                                                         seed = 7)

    # fool_image(train_data_gen)
    # fool_image(train_data_gen2)


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

    image_model = DenseNet169(weights='imagenet', include_top=False)
    # update weights
    out_model = image_model(input)

    # for layer in mobile_net.layers[:-5]:
    #     layer.trainable = False
    # for layer in mobile_net.layers:
    #     print(layer, layer.trainable)



    x = Flatten(name='flatten')(out_model)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(200, activation='sigmoid', name='predictions')(x)

    model = Model(input=input, output=x)
    model.summary()

    lr = 1e-6
    epochs = 10
    model_name = 'dense169_5f'

    # tf.keras.optimizers.adam(...)
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
