import sys
import pathlib
import PIL
import numpy as np
from collections import defaultdict
import os
import shutil
import matplotlib.pyplot as plt

def log(history, name, epoch):
    access_rights = 0o777

    # only have to run once
    model_path = './models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path, access_rights)

    # only have to run once
    logs_path = './logs/'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, access_rights)

    log_path = './logs/' + name + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path, access_rights)

    training_accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    training_loss = history['loss']
    val_loss = history['val_loss']

    x = np.arange(1, epoch + 1)
    plt.figure(1)
    plt.plot(x, [t * 100 for t in training_accuracy])
    plt.plot(x, [v * 100 for v in val_accuracy])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['training','validation'], loc = 'upper left')
    plt.savefig(log_path + 'accuracy.JPEG')

    plt.figure(2)
    plt.plot(x, training_loss)
    plt.plot(x, val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training','validation'], loc = 'upper right')
    plt.savefig(log_path + 'loss.JPEG')
