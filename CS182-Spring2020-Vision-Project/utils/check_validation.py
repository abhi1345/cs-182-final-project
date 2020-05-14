import sys
import pathlib
import PIL
import numpy as np
import os
import shutil


def main():
    with open('eval_classified.csv', 'r') as predicted_file, \
         open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r') as labels_file: #, \
         # open('temp.csv', 'r') as tmp_file: #, \
         # open('eval_fool_val.csv', 'w') as eval_fool_file, \

         # open('to_check.csv', 'w+') as check_file:

         path = './data/tiny-imagenet-200/fool_val/'
         if not os.path.exists(path):
             os.makedirs(path, 0o777)

         predicted_lines = [line.split(',') for line in predicted_file]

         labels_lines = [line.split() for line in labels_file]
         # print(labels_lines)
         total, count = len(predicted_lines), 0

         # temp_lines = [int(line[:len(line)-1]) for line in tmp_file]
         # print(temp_lines)
         # temp_lines.remove(95)
         # temp_lines.remove(99)
         # print(temp_lines)

         # new_labels = []
         # for t in temp_lines:
         #     new_labels += [labels_lines[t]]

         for i in range(total):
             r = len(predicted_lines[i][1])
             # print(labels_lines[i][1], predicted_lines[i][1])
             if labels_lines[i][1] == predicted_lines[i][1][:r-1]:
                 count += 1
         print('Accuracy:', count / total * 100, end='%')

         # f_count = 0
         # for i in range(total):
         #     r = len(predicted_lines[i][1])
         #     if labels_lines[i][1] == predicted_lines[i][1][:r-1]:
         #         # check_file.write('{}\n'.format(labels_lines[i][0]))
         #         if f_count < 100:
         #             source = './data/tiny-imagenet-200/val/images/%s' % labels_lines[i][0]
         #             dest = './data/tiny-imagenet-200/fool_val/'
         #             shutil.copy(source, dest)
         #
                     # num = list(labels_lines[i][0][4:len(labels_lines[i][0])-5])
                     # tmp_file.write('{}\n'.format(''.join(num)))
         #             eval_fool_file.write('{},{},{},{},{}\n'.format(f_count, 'data/tiny-imagenet-200/fool_val/%s' % labels_lines[i][0], 64, 64, 3))
         #             f_count += 1
         #
         #         count += 1
         # print('Accuracy:', count / total * 100, end='%')


if __name__ == '__main__':
    main()
