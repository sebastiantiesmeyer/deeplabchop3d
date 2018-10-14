import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from moviepy.editor import VideoFileClip

from pose_tensorflow.nnet import predict as ptf_predict
from pose_tensorflow.config import load_config
from pose_tensorflow.dataset.pose_dataset import data_to_input


def predict_video(config, video):
    """Predict joint locations for video frames."""

    config_path = Path(config).resolve()
    assert config_path.exists()

    video_path = Path(video).resolve()
    assert video_path.exists()

    project_path = config_path.parents[3]
    training_path = config_path.parent

    print(project_path)
    print(video_path)

    print('Loading test config...')
    cfg = load_config(config_path)

    print('Looking for latest snapshot...')
    snapshots = [s.with_suffix('').name for s in training_path.glob('snapshot-*.index')]
    latest_snapshot_id = max([int(s[len('snapshot-'):]) for s in snapshots])
    latest_snapshot = 'snapshot-{}'.format(latest_snapshot_id)
    snapshot_path = training_path / latest_snapshot
    print('Using snapshot {} at "{}'.format(latest_snapshot_id, snapshot_path))

    cfg['init_weights'] = str(snapshot_path)

    sess, inputs, outputs = ptf_predict.setup_pose_prediction(cfg)

    pdindex = pd.MultiIndex.from_product(
        [['reichler'], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    clip = VideoFileClip(str(video_path))
    n_frames_approx = math.ceil(clip.duration * clip.fps)
    predictions = np.zeros((n_frames_approx, 3 * len(cfg['all_joints_names'])))

    print('Starting pose estimation...')
    for n, frame in enumerate(tqdm(clip.iter_frames(dtype='uint8'), total=n_frames_approx)):
        image_batch = data_to_input(frame)  # skimage.color.gray2rgb(image)
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref = ptf_predict.extract_cnn_output(outputs_np, cfg)
        pose = ptf_predict.argmax_pose_predict(scmap, locref, cfg.stride)
        predictions[n, :] = pose.flatten()

    print('Storing results')
    df = pd.DataFrame(predictions[:n, :], columns=pdindex, index=range(n))
    df.to_hdf(video_path.with_suffix('.h5'), 'df_with_missing', format='table', mode='w')
    df.to_csv(video_path.with_suffix('.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('config')

    cli_args = parser.parse_args()
