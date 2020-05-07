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

    with open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r') as labels_file:
        labels_lines = [line.split()[0] for line in labels_file]

    return parts[-1] == labels_lines

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img)

    return img, label

def main():
    pretrained_model = tf.keras.models.load_model('./models/mobilenet_10.h5')
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    val_dir = pathlib.Path('./data/tiny-imagenet-200/val/images/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])

    with open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r') as labels_file:
        labels_lines = [line.split() for line in labels_file]

    list_ds = tf.data.Dataset.list_files(str(val_dir/'*'), shuffle=False)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    for image, label in labeled_ds:
        start_time = time.time()
        image = preprocess(image)
        s = np.argmax(label.numpy())

        label = tf.one_hot(CLASSES.index(labels_lines[s][1]), len(CLASSES))
        label = tf.reshape(label, (1, len(CLASSES)))

        perturbations = create_adversarial_pattern(image, label, pretrained_model)
        eps = 0.1
        adv_x = image + eps * perturbations
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        image = adv_x.numpy()
        matplotlib.image.imsave('./data/tiny-imagenet-200/new_val/val_{}.JPEG'.format(s), image[0])

        print('took:', time.time() - start_time)

if __name__ == '__main__':
    main()
