import pathlib
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import time
import matplotlib

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (64, 64))
    image = image[None, ...]
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)
    return image

def create_adversarial_pattern(input_image, input_label, pretrained_model):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)

    signed_grad = tf.sign(gradient)
    return signed_grad


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    return parts[-1]

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img)

    return img, label

def main():
    pretrained_model = tf.keras.models.load_model('./models/mobilenet_10.h5')
    data_dir = pathlib.Path('./data/tiny-imagenet-200/new_train/')

    CLASSES = sorted([item.name for item in data_dir.glob('*')])

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpeg'), shuffle=False)

    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    time_elapsed = []
    for image, label in labeled_ds:
        start_time = time.time()

        label = str(label.numpy())
        name = label.split('.')[0][2:]
        class_name = label.split('_')[0][2:]
        true_label = CLASSES.index(class_name)


        image = preprocess(image)

        label = tf.one_hot(true_label, len(CLASSES))
        label = tf.reshape(label, (1, len(CLASSES)))

        perturbations = create_adversarial_pattern(image, label, pretrained_model)
        eps = 0.1
        adv_x = image + eps * perturbations
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        image = adv_x.numpy()
        matplotlib.image.imsave('./data/tiny-imagenet-200/new_train/{}/images/{}.JPEG'.format(class_name, name + 'f'), image[0])

        end_time = time.time() - start_time
        time_elapsed += [end_time]
        if len(time_elapsed) % 100 == 0:
            print('\ntotal time so far:', sum(time_elapsed))
            print('fooled images created:', len(time_elapsed))

    print('\ntotal time: ', sum(time_elapsed))

if __name__ == '__main__':
    main()
