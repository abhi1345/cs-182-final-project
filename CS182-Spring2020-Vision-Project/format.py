import sys
import pathlib
import PIL
import numpy as np
from collections import defaultdict
import os
import shutil


def main():
    with open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r') as val_file:

        val_lines = [line.split() for line in val_file]
        total, count = len(val_lines), 0

        lst = [(line[1], line[0]) for line in val_lines]
        d = defaultdict(list)
        for k,v in lst:
            d[k].append(v)
        new_lst = sorted(d.items())
        path = os.getcwd()

        for row in new_lst:
            path = './data/tiny-imagenet-200/validation/%s/images' % row[0]
            access_rights = 0o777
            if not os.path.exists(path):
                os.makedirs(path, access_rights)

            class_name, images = row[0], row[1]

            for f in images:
                source = './data/tiny-imagenet-200/val/images/%s' % f
                dest = './data/tiny-imagenet-200/validation/%s/images/' % class_name
                shutil.copy(source, dest)

if __name__ == '__main__':
    main()
