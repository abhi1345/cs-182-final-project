import sys
import pathlib
import PIL
import numpy as np

def main():
    # Load the classes
    val_path = 'data/tiny-imagenet-200/val/images'
    test_path = 'data/tiny-imagenet-200/test/images'
    val_dir = pathlib.Path('./' + val_path)
    test_dir = pathlib.Path('./' + test_path)

    val_image_count = len(list(val_dir.glob('**/*.JPEG')))
    test_image_count = len(list(test_dir.glob('**/*.JPEG')))
    val_image_count_small = 1000

    print('Discovered {} val_images'.format(val_image_count))
    print('Discovered {} test_images'.format(test_image_count))

    # Loop through the CSV file and make a prediction for each line
    with open('eval_val.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for i in range(val_image_count):
            eval_output_file.write('{},{},{},{},{}\n'.format(i, val_path + '/val_' + str(i) + '.JPEG', 64, 64, 3))

    # with open('eval_val_small.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
    #     for i in range(val_image_count_small):
    #         eval_output_file.write('{},{},{},{},{}\n'.format(i, val_path + '/val_' + str(i) + '.JPEG', 64, 64, 3))

    with open('eval_test.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for i in range(test_image_count):
            eval_output_file.write('{},{},{},{},{}\n'.format(i, test_path + '/val_' + str(i) + '.JPEG', 64, 64, 3))

if __name__ == '__main__':
    main()
