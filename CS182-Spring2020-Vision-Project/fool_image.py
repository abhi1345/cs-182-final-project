import pathlib
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (64, 64))
    image = image[None, ...]
    return image

def create_adversarial_pattern(input_image, input_label):
    pretrained_model = tf.keras.models.load_model('./models/dense121_5.h5')
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

def display_images(image, description, CLASSES, eps):
    pretrained_model = tf.keras.models.load_model('./models/dense121_5.h5')
    image_probs = pretrained_model.predict(image)
    prediction = int(np.argmax(image_probs.reshape(-1)))
    plt.figure()
    plt.imshow(image[0])
    plt.title('{} | {} \n {}'.format(prediction, CLASSES[prediction], description))
    path = './data/tiny-imagenet-200/new_val/val_0_{}.JPEG'.format(eps)
    plt.savefig(path)


def main():
    pretrained_model = tf.keras.models.load_model('./models/dense121_5.h5')
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])

    # image_path = tf.keras.utils.get_file('data/tiny-imagenet-200/test/images/val_0.JPEG')
    image_raw = tf.io.read_file('./data/tiny-imagenet-200/val/images/val_0.JPEG')
    image = tf.image.decode_image(image_raw)

    image = preprocess(image)
    # plt.figure()
    # plt.imshow(image[0])

    image_probs = pretrained_model.predict(image)
    prediction = int(np.argmax(image_probs.reshape(-1)))
    # _, image_class, class_confidence = get_imagenet_label(image_probs)
    # plt.title('{} | {}'.format(prediction, CLASSES[prediction]))

    # plt.show()


    labrador_retriever_index = prediction
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)
    # plt.imshow(perturbations[0])
    # plt.show()

    epsilons = [0, 0.01, 0.1, 0.15]
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

    for i, eps in enumerate(epsilons):
      adv_x = image + eps*perturbations
      adv_x = tf.clip_by_value(adv_x, 0, 1)
      display_images(adv_x, descriptions[i], CLASSES, eps)
      # tf.io.write_file('./data/tiny-imagenet-200/new_val/')
      # plt.savefig(os.path.join('./data/tiny-imagenet-200/new_val/', adv_x))


# def fool_image(image_gen):
#     model = tf.keras.models.load_model('./models/mobile_test_3.h5')
#     loss_object = tf.keras.losses.CategoricalCrossentropy()
#
#     dict = image_gen.class_indices
#
#     key_dict = list(dict.keys())
#
#     loss = tf.constant(5.2441)
#
#     x, y = image_gen.next()
#     input_image = tf.cast(x, tf.float32)
#     print("\n\n\n\n\n\n",tf.shape(input_image))
#     input_image = tf.reshape(input_image, (32, 64, 64, 3))
#     # print(loss)
#     print("\n\n\n\n\n\n",tf.shape(input_image))
#     # for i in range(0,1):
#     #     input_image = x[i]
#         # print("\n\n\n\n", key_dict[np.argwhere(y[i])[0]])
#         # plt.imshow(input_image)
#         # plt.show()
#
#     # Generate a prediction
#     prediction_probs = model.predict(np.expand_dims(input_image[0], 0))
#     label = int(np.argmax(prediction_probs.reshape(-1)))
#     index = np.argmax(y[0])
#     label = tf.one_hot(index, prediction_probs.shape[-1])
#
#     label = tf.reshape(label, (1, prediction_probs.shape[-1]))
#     print("\n\n\n\n", label)
#
#     with tf.GradientTape() as tape:
#         tape.watch(input_image)
#         prediction = model(x[0])
#         loss = loss_object(label, prediction)
#
#       # Get the gradients of the loss w.r.t to the input image.
#     gradient = tape.gradient(loss, input_image)
#     print("\n\n\n\n", gradient)
#   # Get the sign of the gradients to create the perturbation
#     signed_grad = tf.sign(gradient)
#     plt.show(signed_grad[0])
#     return signed_grad

if __name__ == '__main__':
    main()
