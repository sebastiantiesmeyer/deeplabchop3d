import os
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from skimage import draw, io
from tqdm import tqdm
import matplotlib.pyplot as plt

DEFAULT_COLORMAP = 'RdYlGn'


def label_csv_to_dict(csv_path):
    """Convert a minimized csv of labels into a dictionary with image file path as key.
    """
    with open(csv_path) as f:
        csv_reader = csv.reader(f)
        rows = list(csv_reader)

    img_d = defaultdict(list)
    for row in rows:
        img_d[row[0]].append(row[1:])
    return img_d


def reduce_imagej_csv(path, parts, scorer=None, split=False, full_joint_name=True):
    """Read a (typically) `Results.csv` file created by the ImageJ `Measure` tool and store as
    a minimal csv. The ImageJ csv must be in the same directory as the images it was created from.
    Matches the measurements to the images in the directory.

    In:
        path: Path to image set or measurements csv file
        parts: List of names of body parts in order they were labeled in ImageJ
        scorer: Name of scorer. If none, will take from platform
        split: bool, split into multiple csv files, one per body part
        full_joint_name: bool, show joint identity as string or as 0-indexed integer

    Out: csv file in format [imgXXX.png], [bodypart], [x], [y]
    """
    # If no scorer name is provided, take currently logged in user name. Accountability, yo.
    if scorer is None:
        scorer = os.getlogin()
        raise Warning('Scorer not provided. Assuming scoring by user "{}".'.format(scorer))

    # Figure out if input path is the results file from ImageJ, or a directory containing it and images.
    csv_path = next(path.glob('Results.csv')) if path.is_dir() else path
    path = csv_path.parent

    # Images are assumed to be in "img[0*d].png" format
    # Note platform depending case sensitivity of glob. Looking at you, MS Paint.
    images = sorted([img.name for img in path.glob('img*.png')])
    with open(csv_path) as csv_file:
        labels = csv_file.readlines()[1:]
    print('Read "{}"'.format(csv_path))

    # Check that we have exactly the right amount of labels for all images
    if len(parts) * len(images) != len(labels):
        raise ValueError('Number of labels and images does not match '
                         '({} images != {} labels / {} bodyparts)!'.format(len(images), len(labels), len(parts)))
    else:
        print('{} images in directory, {} labels per {} joints.'.format(len(images), len(labels), len(parts)))

    # Clobber image names, scorer tag, body parts and label coordinates together
    rows = []
    for n, label in enumerate(labels):
        image_idx = n // len(parts)
        bodypart_id = n % len(parts)

        # ImageJ has sub-pixel precision, which we don't need, and we'll strip out the noisy 0s, too.
        x, y = map(lambda x: str(int(float(x))), label.split(',')[5:7])

        # We can identify the joints by their full name, or their index in the list of joints
        # For anything before the actual training step, human readable joint names are nice for
        # development and verification/fixing missing labels, etc.
        joint = str(parts[bodypart_id] if full_joint_name else bodypart_id)
        rows.append([images[image_idx], scorer, joint, x, y])

    # Write to single or part specific csv file
    fname_suffix = parts if split else ['multijoint']
    for n, suffix in enumerate(fname_suffix):
        csv_path_part = path / Path(suffix + '.csv')
        with open(csv_path_part, 'w') as f:
            # every nth row to crudely filter by body part
            # assuming they are in order
            for row in rows[n::len(fname_suffix)]:
                f.write(','.join(row) + '\n')
        print('Wrote "{}"'.format(str(csv_path_part)))

    return rows

def draw_image_labels(csv_file, joints, cmap_name=None):
    csv_file = Path(csv_file).resolve()
    parent_path = Path(csv_file.parent)

    if cmap_name is None:
        cmap_name = DEFAULT_COLORMAP
    cmap = plt.cm.get_cmap(cmap_name, len(joints))

    # Path to directory containing the individual video data sets and their labels
    #
    # labels_cmap = plt.cm.get_cmap(cfg.colormap, len(cfg.bodyparts))
    part_ids = {part: part_id for part_id, part in enumerate(joints)}

    # # Make list of different video data sets:
    # datasets = [d for d in data_path.glob('*') if d.is_dir() and 'labeled' not in d.name]
    # print('Found {} dataset(s) to label in "{}"'.format(len(datasets), data_path))

    # directory for the labeled images
    labeled_path = parent_path / 'labeled'
    if not labeled_path.exists():
        print('Created "{}" for labeled images'.format(labeled_path))
        labeled_path.mkdir()

    # map csv to dictionary for easy batching of labeling. That way we
    # only need to iterate the keys and draw all labels together
    print('Loading csv {}'.format(csv_file))
    img_d = label_csv_to_dict(csv_file)

    # Draw labels into all images
    for img_name, labels in tqdm(img_d.items()):
        image = io.imread(parent_path / img_name)
        for lbl in labels:
            # draw colored marker on label position
            cx, cy, radius = float(lbl[-1]), float(lbl[-2]), 2
            rr, cc = draw.circle(cx, cy, radius, image.shape)
            try:
                col_idx = int(lbl[1])
            except ValueError:
                col_idx = part_ids[lbl[1]]
            color = [int(cv * 255) for cv in cmap(col_idx)[:3]]
            image[rr, cc] = color

        # save labeled image
        io.imsave(labeled_path / img_name, image)


def combine_labels(data_path):
    """Gather minimized labels files from all datasets of a project
    """
    datasets = [d for d in data_path.glob('*') if d.is_dir() and 'labeled' not in d.name]
    parents = list({d.parent for d in datasets})
    assert(len(parents)) == 1

    # Haha, get it?
    destination = parents[0] / 'joint_labels.csv'

    img_dict_all = {}
    for ds in datasets:
        joint_file = ds / 'multijoint.csv'
        print('Reading {}'.format(joint_file))
        img_dict = label_csv_to_dict(joint_file)

        # Update image path to include parent name
        # The individual files are relative to their position within the
        # image set directory. We'll place the combined file into the parent.
        img_dict_up = {ds.parts[-1] + '/' + img_path: labels for img_path, labels in img_dict.items()}
        img_dict_all.update(img_dict_up)

    # Write csv file with labels for all image sets
    with open(destination, 'w') as f:
        for img_path, img_labels in img_dict_all.items():
            for lbl in img_labels:
                f.write(','.join([img_path, *lbl]) + '\n')

    print('Wrote combined labels file "{}"'.format(destination))


def draw_labels(image, labels, parts, cmap_name=None, msize=1):
    """Draw labels into an image."""
    if cmap_name is None:
        cmap_name = DEFAULT_COLORMAP
    cmap = plt.cm.get_cmap(cmap_name, len(parts))

    for lbl in labels:
        # draw colored marker on label position
        col_idx = int(lbl[1])
        cx, = float(lbl[-1])
        cy = float(lbl[-2])
        radius = msize

        rr, cc = draw.circle(cx, cy, radius, image.shape)
        color = [int(cv * 255) for cv in cmap(col_idx)[:3]]

        image[rr, cc] = color

    return image


def prepare_mat_array(img_dict, data_path, cfg, img_path_prefix=''):
    """Convert dict of labels for a set of training images to array
    to be converted to a .mat file for DeeperCut.
    """
    dist = lambda x, y: math.sqrt(x ** 2 + y ** 2)
    joint_ids = {j_name: j_id for j_id, j_name in enumerate(cfg.bodyparts)}

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
        joint_list = [j for j in joint_list if dist(j[1], j[2]) > cfg.invisibleboundary]

        # warn if label outside of image size
        for j in joint_list:
            if j[1] >= img_w or j[2] >= img_h:
                raise ValueError('Label outside image size for image "{}"'.format(image_path))

        joints = np.zeros((1, 1), dtype='O')
        joints[0, 0] = np.array(joint_list)
        ds_arr['joints'][0, n] = joints

    return ds_arr