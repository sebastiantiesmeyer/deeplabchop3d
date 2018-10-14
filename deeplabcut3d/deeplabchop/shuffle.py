import os
import yaml
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm, trange

from deeplabchop import label
from deeplabchop.util import prepare_mat_array, label_dict_to_csv


def shuffle(csv_file, train_fraction, destination, joints, boundary):
    csv_file = Path(csv_file).resolve()
    destination = Path(destination)
    # if not destination.exists():
    #     destination.mkdir(parents=True)
    #     print('Created {}'.format(destination))

    img_dict = label.label_csv_to_dict(csv_file)
    img_keys = np.random.permutation(list(img_dict.keys()))

    cutoff = int(len(img_keys) * train_fraction)

    img_keys_train = img_keys[:cutoff]
    img_keys_test = img_keys[cutoff:]

    img_dict_train = {key: img_dict[key] for key in img_keys_train}

    # Currently deepcut needs to have the mat file point relative to the model training site,
    # with the whole structure just ... odd.
    rel_img_path = str(Path('../../../data/').as_posix()) + '/' # .joinpath(csv_file.parts[-1])
    mat_arr = prepare_mat_array(img_dict_train, csv_file.parent, joints, boundary, rel_img_path)

    # Write .mat file
    mat_path = Path(destination) / 'training.mat'
    sio.savemat(mat_path, {'dataset': mat_arr})

    # Write .csv file
    label_dict_to_csv(img_dict_train, mat_path.with_suffix('.csv'))

    tqdm.write('Saved labels for {} training images to "{}"'.format(len(img_dict_train), mat_path))


def training_pose_yaml(in_file, updates, out_file):
    with open(in_file) as yf:
        raw = yf.read()

    docs = []
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc))
        except SyntaxError:
            docs.append(raw_doc)

    for key, value in updates.items():
        docs[0][key] = value

    with open(out_file, 'w') as f:
        yaml.dump(docs[0], f)
    return docs[0]


def test_pose_yaml(in_dict, keys, out_file):
    test_dict = {key: in_dict[key] for key in keys}
    test_dict['scoremap_dir'] = 'test'

    with open(out_file, 'w') as yf:
        yaml.dump(test_dict, yf)