import os
import shutil
import yaml
from pathlib import Path
from datetime import datetime as dt

from deeplabchop import DEBUG
from deeplabchop.util import update_yaml


YAML_KEY_COMMENTS = {
    'experimenter': '\n# Name of experimenter/scorer. Keep to alphanumeric characters.\n',
    'image_sets': '\n# List of paths to the image sets generated from videos.\n',
    'num_frames': '\n# Number of frames to extract per video...\n',
    'joints': '\n# List of joints to be labeled and tracked. The order must match to annotations.\n',
    'num_shuffles': '\n# Number of shuffles to perform per training session\n',
    'train_fraction': '\n# Fraction of images to use for training.\n',
    'random_seed': '\n# Starting seed used to initialize the numpy random number generator for shuffling.\n ' +
                   '# This should allow some reproducibility. The seed used for each shuffle is\n' +
                   '# [random_seed] + [shuffle_id].\n',
    'cmap': '\n# (Matplotlib) color map name. Best not too dark and non-circular.',
    'boundary': '\n# Distance from top and right of image where labels will count as rejected\n'
}


def yaml_config_template(yaml_path, cfg):
    """Write a dictionary of configuration into a new yaml file.
    Group keys thematically belong together and add comments.
    """
    # TODO: This should be based on a collections.defaultdict
    with open(yaml_path, 'w') as cf:
        keys = list(cfg.keys())

        # known keys:
        known_keys = YAML_KEY_COMMENTS
        cf.write('# DeepLabChop Project configuration\n')
        cf.write('# Paths are in in Posix form (./path/to/dir/file), relative to the project directory\n')
        for k in known_keys:
            if k in keys:
                cf.write(YAML_KEY_COMMENTS[k])
                yaml.dump({k: cfg[k]}, cf, default_flow_style=False)
                keys.remove(k)

        # unknown keys:
        cf.write('\n# Unknown keys\n')
        while len(keys):
            k = keys.pop()
            yaml.dump({k: cfg[k]}, cf, default_flow_style=False)


def project(project, experimenter, videos, working_directory=None, copy_videos=True):
    """Create a new project directory, sub-directories and basic configuration file.
    """
    date = dt.today().strftime('%Y-%m-%d')
    wd = Path(working_directory).resolve()
    project_name = '{pn}-{exp}-{date}'.format(pn=project, exp=experimenter, date=date)
    project_path = wd / project_name

    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return

    # video contains raw videos, data the image sets and results... results.
    video_path = project_path / 'videos'
    data_path = project_path / 'data'
    shuffles_path = project_path / 'shuffles'
    results_path = project_path / 'results'
    for p in [video_path, data_path, shuffles_path, results_path]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    # Unless specified not to, copy the videos given into the project/video directory.
    # Update video path list to the new location relative to the project base path.
    videos = [Path(vp) for vp in videos]
    if copy_videos:
        destinations = [video_path.joinpath(vp.name) for vp in videos]
        for src, dst in zip(videos, destinations):
            if dst.exists() and not DEBUG:
                raise FileExistsError('Video {} exists already!'.format(dst))
            try:
                shutil.copy2(src, dst)
            except shutil.SameFileError:
                if not DEBUG:
                    raise

            print('Copied {} to {}'.format(src, dst))
            videos = destinations

    # One image set per video
    # The configuration file should store all paths relative to the project base directory
    # This requires some ugly path wrangling, but hopefully pays off later
    image_sets = {}
    for video in videos:
        rel_video_path = Path(os.path.relpath(video, project_path)).as_posix()
        rel_img_path = Path(os.path.relpath(data_path, project_path)).joinpath(video.with_suffix('').name).as_posix()

        image_sets[str(rel_video_path)] = {'img_path': rel_img_path,
                                           'crop': ', '.join(map(str, [0, 0, -1, -1]))}

    # Configuration file templates
    cfg_dict = {'experimenter': experimenter,
                'num_frames': 100,
                'joints': ['beak', 'dorsal_fin', 'bee_knee'],
                'num_shuffles': 3,
                'image_sets': image_sets,
                'random_seed': 0,
                'cmap': 'RdYlGn',
                'train_fraction': 0.95,
                'boundary': 10}

    # Write dictionary to new yaml file
    # Will write comments and group keys in a way that is not completely random.
    yaml_config_template(project_path / 'config.yaml', cfg_dict)
    print('Generated "{}"'.format(project_path / 'config.yaml'))

    # Create status yaml file
    update_yaml(project_path / 'status.yaml', {'Created': dt.today().strftime('%Y-%m-%d_%H-%M-%S')},
                create=True, empty=True)
