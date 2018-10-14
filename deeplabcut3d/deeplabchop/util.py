import csv
import math
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
from skimage import io

from deeplabchop import DEBUG


def read_yaml(yaml_path):
    if not yaml_path.exists():
        raise FileNotFoundError('No such file or directory: {}'.format(yaml_path))

    yd = dict()
    with open(yaml_path, 'r') as yf:
        yd.update(yaml.load(yf))

    return yd


def update_yaml(yaml_path, status_dict, create=False, empty=False):
    """Update the `status.yaml` file for a project."""
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        if not create:
            raise FileNotFoundError('"{}" not found.'.format(yaml_path))
        else:
            yaml_path.touch()
            print('Created {}'.format(yaml_path))

    if not empty:
        with open(yaml_path) as yf:
            yml_dict = yaml.load(yf)
            yml_dict = {} if yml_dict is None else yml_dict
    else:
        yml_dict = {}

    yml_dict.update(status_dict)

    with open(yaml_path, 'w+') as yf:
        yaml.dump(yml_dict, yf, default_flow_style=False)

    if DEBUG:
        print('Updated yml_dict:', yml_dict)


def label_csv_to_dict(csv_path):
    """Convert a csv of labels into a dictionary with image file path as key.
    """
    with open(csv_path) as f:
        csv_reader = csv.reader(f)
        rows = list(csv_reader)

    img_d = defaultdict(list)
    for row in rows:
        img_d[row[0]].append(row[1:])
    return img_d

def label_dict_to_csv(label_dict, csv_path):
    """Convert dictionary of labels to csv file."""
    with open(csv_path, 'w') as cf:
        for img_path in sorted(label_dict.keys()):
            for lbl in label_dict[img_path]:
                cf.write(','.join(map(str, [img_path, *lbl]))+'\n')

def dist(x, y):
    """Length of 2D vector"""
    return math.sqrt(x ** 2 + y ** 2)


def prepare_mat_array(img_dict, data_path, joints, boundary, img_path_prefix=''):
    """Convert dict of labels for a set of training images to array
    to be converted to a .mat file for DeeperCut.
    """
    joint_ids = {j_name: j_id for j_id, j_name in enumerate(joints)}

    # base array structure
    ds_arr = np.zeros((1, len(img_dict)), dtype=[('image', 'O'), ('size', 'O'), ('joints', 'O')])
    ds_arr['joints'] = np.zeros((len(img_dict),), dtype='O')

    # The np.array -> matlab struct array/cell array mapping is no fun to work with
    for n, (image_path, labels) in enumerate(img_dict.items()):
        # Image metadata (path and shape)
        image = io.imread(data_path / image_path)
        img_h, img_w, num_colors = image.shape

        ds_arr[0, n]['image'] = np.array([img_path_prefix + image_path], dtype='U')
        ds_arr[0, n]['size'] = np.array([[num_colors, img_h, img_w]], dtype='uint16')

        # joint label identity either 0-indexed integer, or a name as string
        # for the mat file we need the integer
        try:
            joint_list = [[int(float(v)) for v in lbl[-3:]] for lbl in labels]
        except ValueError:
            joint_list = [[joint_ids[lbl[-3]], int(float(lbl[-2])), int(float(lbl[-1]))] for lbl in labels]

        # drop joint labels that are too close to origin, signaling
        # experimenter marked joint as invisible in image
        joint_list = [j for j in joint_list if dist(j[1], j[2]) > boundary]

        # warn if label outside of image size
        for j in joint_list:
            if j[1] >= img_w or j[2] >= img_h:
                raise ValueError('Label outside image size for image "{}"'.format(image_path))

        joints_arr = np.zeros((1, 1), dtype='O')
        joints_arr[0, 0] = np.array(joint_list)
        ds_arr['joints'][0, n] = joints_arr

    return ds_arr
