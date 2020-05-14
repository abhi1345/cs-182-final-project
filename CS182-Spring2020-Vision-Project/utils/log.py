import sys
import pathlib
import PIL
import numpy as np
from collections import defaultdict
import os
import shutil
import matplotlib.pyplot as plt
import keras
import time
import datetime
import pandas as pd

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def cparams(csv_path, name, lr, epochs, time, val_acc, top_k_acc):
    df = pd.DataFrame(np.array([[name.ljust(11, ' '), lr, epochs, time, val_acc, top_k_acc]]), columns=['Model Name', 'lr','Epochs', 'Total Time', 'Val_Acc', 'Top 5'])
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, sep='\t')

    else:
        df_og = pd.read_csv(csv_path, sep='\t', index_col=0)

        appended = df.append(df_og, ignore_index=True)
        appended['Top 5'] = appended['Top 5'].astype('float')

        result = appended.sort_values(by='Top 5', ascending=False)
        result.to_csv(csv_path, sep='\t')

def log(history, logged_params):

    # only have to run once
    model_path = './models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path, access_rights)

    # only have to run once
    logs_path = './logs/'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, access_rights)

    name = logged_params['name']
    lr = logged_params['lr']
    epochs = logged_params['epochs']
    time = str(datetime.timedelta(seconds=logged_params['time'])) + '  '
    training_accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    training_loss = history['loss']
    val_loss = history['val_loss']
    top_5_acc = history['top_k_categorical_accuracy']
    val_top_5_acc = history['val_top_k_categorical_accuracy']


    cparams('./logs/tests.csv', name, lr, epochs, time, round(val_accuracy[-1] * 100, 2), round(val_top_5_acc[-1] * 100, 2))

    log_path = './logs/' + name + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path, 0o777)


    x = np.arange(1, epochs + 1)
    plt.figure(1)
    plt.plot(x, [t * 100 for t in training_accuracy])
    plt.plot(x, [v * 100 for v in val_accuracy])
    plt.plot(x, [t5 * 100 for t5 in top_5_acc])
    plt.plot(x, [v5 * 100 for v5 in val_top_5_acc])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['training','validation', 'top5_training', 'top5_val'], loc = 'upper left')
    plt.savefig(log_path + 'accuracy.JPEG')

    plt.figure(2)
    plt.plot(x, training_loss)
    plt.plot(x, val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training','validation'], loc = 'upper right')
    plt.savefig(log_path + 'loss.JPEG')
