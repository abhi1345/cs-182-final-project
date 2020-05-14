import sys
import pathlib
import PIL
import numpy as np
import shutil
import os
import random
import collections

def main():
    # Load the classes
    val_path = 'data/tiny-imagenet-200/val/images'
    test_path = 'data/tiny-imagenet-200/test/images'
    val_dir = pathlib.Path('./' + val_path)
    new_val_dir = pathlib.Path('./data/tiny-imagenet-200/new_val')
    test_dir = pathlib.Path('./' + test_path)

    val_image_count = len(list(val_dir.glob('**/*.JPEG')))
    test_image_count = len(list(test_dir.glob('**/*.JPEG')))
    val_image_count_small = 1000

    print('Discovered {} val_images'.format(val_image_count))
    print('Discovered {} test_images'.format(test_image_count))

    # Loop through the CSV file and make a prediction for each line
    # with open('s.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
    #     for i in range(val_image_count):
    #         eval_output_file.write('{},{},{},{},{}\n'.format(i, 'data/tiny-imagenet-200/new_val' + '/val_' + str(i) + '.JPEG', 64, 64, 3))

    # only have to run once
    # new_train_path = './data/tiny-imagenet-200/new_train/'
    # if not os.path.exists(new_train_path):
    #     os.makedirs(new_train_path, 0o777)

    # data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    # CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    #
    # for c in CLASS_NAMES:
    #     for i in range(500):
    #         source = './data/tiny-imagenet-200/fool_train/{}/images/{}'.format(c, c + '_' + str(i) + 'f.JPEG')
    #         dest = './data/tiny-imagenet-200/b_train/{}/images/'.format(c)
    #         shutil.copy(source, dest)

    # with open('eval_fv.csv', 'w') as fool_val_file:
    #     random.seed(7)
    #     total = range(10000)
    #     first_half = random.sample(total, 5000)
    #     second_half = list(set(total).difference(set(first_half)))
    #
    #     for i in total:
    #         if i in first_half:
    #             fool_val_file.write('{},{},{},{},{}\n'.format(i, 'data/tiny-imagenet-200/val/images' + '/val_' + str(i) + '.JPEG', 64, 64, 3))
    #         else:
    #             fool_val_file.write('{},{},{},{},{}\n'.format(i, 'data/tiny-imagenet-200/new_val' + '/val_' + str(i) + '.JPEG', 64, 64, 3))

    # path = './data/tiny-imagenet-200/fool_val'
    # if not os.path.exists(path):
    #     os.makedirs(path, 0o777)
    #
    # random.seed(7)
    # total = range(10000)
    # first_half = random.sample(total, 5000)
    # second_half = list(set(total).difference(set(first_half)))
    #
    # for i in total:
    #     if i in first_half:
    #         shutil.copy('./data/tiny-imagenet-200/val/images/val_' + str(i) + '.JPEG', path)
    #
    #     else:
    #         shutil.copy('./data/tiny-imagenet-200/new_val/val_' + str(i) + '.JPEG', path)


    # with open('s.csv', 'w+') as new_val_file, \
    #      open('temp.csv', 'r') as temp_file:
    #      temp_lines = [line[:len(line)-1] for line in temp_file]
    #     # s  = list(new_val_dir.glob('**/*.JPEG'))
    #      for i in range(len(temp_lines)):
    #          new_val_file.write('{},{},{},{},{}\n'.format(i, 'data/tiny-imagenet-200/new_val' + '/val_' + temp_lines[i] + '.JPEG', 64, 64, 3))
            # new_val_file.write('{},{},{},{},{}\n'.format(i, str(s[i]).replace("\\","/"), 64, 64, 3))

    # with open('as.csv', 'w+') as new_val_file, \
    #      open('temp.csv', 'r') as temp_file:
    #      temp_lines = [line[:len(line)-1] for line in temp_file]
    #      # print(temp_lines)
    #      # s  = list(new_val_dir.glob('**/*.JPEG'))
    #      # s = [str(es) for es in s]
    #      # # print(s)
    #      # d = [es.split('\\')[3] for es in s]
    #
    #      for i in range(len(temp_lines)):
    #          new_val_file.write('{},{},{},{},{}\n'.format(i, val_path + '/val_' + temp_lines[i] + '.JPEG', 64, 64, 3))


    # with open('eval__fool_val.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
    #     for i in range(val_image_count_small):
    #         eval_output_file.write('{},{},{},{},{}\n'.format(i, val_path + '/val_' + str(i) + '.JPEG', 64, 64, 3))

    with open('eval_test.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for i in range(test_image_count):
            eval_output_file.write('{},{},{},{},{}\n'.format(i, test_path + '/test_' + str(i) + '.JPEG', 64, 64, 3))

if __name__ == '__main__':
    main()
