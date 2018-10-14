import yaml
from pathlib import Path
import pkg_resources

from tqdm import trange, tqdm
from deeplabchop import DEBUG, util, status, extract, label, shuffle, GUI_utils#, training


def _echo(s):
    # poor man's module specific logging
    print('Wizard: ' + s)


def step(project):
    # TODO: Allow returning to specific points in the wizard without having to edit status.yml
    project = Path(project)
    project_status = status.read_status(project)
    if DEBUG:
        status.show_status(project)

    cfg = util.read_yaml(project / 'config.yaml')

    # Control for succesful project creation
    # -------------------------------------------------------------------------------------------
    if 'Created' not in project_status:  # sanity check
        _echo("Whoopsie, this ain't Kansas no more! Something went terribly wrong during project creation.")
    else:
        _echo('Project creation OK')

    # Check that dataset list in config file and in data directory match
    # -------------------------------------------------------------------------------------------
    data_path = project / 'data'
    datasets = [d for d in data_path.glob('*') if d.is_dir() and 'labeled' not in d.name]
    if len(datasets) != len(cfg['image_sets']):
        Warning('Dataset directory not matching configuration file image set list.')
        # a lot of string wrangling just to have the output not be too confusing...
        print('Config:', ''.join(['\n\t' + k['img_path'] for k in cfg['image_sets'].values()]))
        print('data  :', ''.join(['\n\t' + '/'.join(k.parts[-2:]) for k in datasets]))
        print('')
        # TODO: Update command to add new image sets to existing project

    # Check cropping configuration for videos
    # -------------------------------------------------------------------------------------------
    if 'CropAdjust' not in project_status or not project_status['CropAdjust']:
        _echo('Checking video cropping configuration...')

        # Create directory for image sets
        if not len(cfg['image_sets']):
            print('No videos given to create image sets. Stopping.')
            return

        for video, metadata in cfg['image_sets'].items():
            video_path = project / video
            img_path = project / 'data' / video_path.with_suffix('').name
            img_path.mkdir(exist_ok=True)

            # check cropping
            crop = list(map(int, metadata['crop'].split(',')))
            extract.crop_examples(video_path, crop, img_path)
            _echo('Check cropping of "{}" in "{}"'.format(video, img_path))
        util.update_yaml(project / 'status.yaml', {'CropAdjust': True})
        _echo('Check image set directories for original and cropped example frames and '
              'adjust as needed.\n---------------------------------------------------')
        return
    else:
        _echo('Crop adjustment OK')

    # Extract images from all videos
    # -------------------------------------------------------------------------------------------
    if 'FrameExtraction' not in project_status or not project_status['FrameExtraction']:
        _echo('Next up: Creating image sets...')
        num_frames = max(2,int(cfg['num_frames']) // len(cfg['image_sets']))

        # Loop over all video files specified in the project configuration
        # Total number of frames is evenly divided between the videos.
        # Does not take length of the video into account... worse, a very short
        # video might have duplicates extracted!
        # TODO: Distribute extracted frames according to video length

        seed = int(cfg['random_seed']) #+ n

        extract.extract_frames(cfg['image_sets'], video_root=project,
                                num_frames=num_frames,
                                destination=project/'data/ExtractedFrames.h5',
                                seed=seed)

        # for n, (video, metadata) in enumerate(cfg['image_sets'].items()):
        #     Path.mkdir(project / metadata['img_path'])
        #     video_path = Path(video)
            
            
        #     print('Extracting {} frames with seed {} from {}'.format(num_frames, seed, video_path.name))


        util.update_yaml(project / 'status.yaml', {'FrameExtraction': True})
        print('\nYou can now annotate frames in e.g. ImageJ and store the results in the '
              'image set directories as `Results.csv`.')
        return

    else:
        _echo('Frame Extraction OK')

    # User annotation of joint position in image set
    # -------------------------------------------------------------------------------------------
    if 'Annotated' not in project_status or not project_status['Annotated']:
        do_labeling = input('Run labeling GUI? [Y/n]')
        if not do_labeling in ['N','n']:
            GUI_utils.run_labeler(cfg,root=project)

        util.update_yaml(project / 'status.yaml', {'Annotated': True})
        _echo('Minimized {} image set label file(s).'.format(len(cfg['image_sets'])))
    else:
        # TODO: Check existence of csv files
        _echo('Image set annotations OK')

    # # Draw labels on images for verification
    # # -------------------------------------------------------------------------------------------
    # if 'TrainingLabelsDrawn' not in project_status or not project_status['TrainingLabelsDrawn']:
    #     _echo('Drawing labels on images in data sets for verification...')
    #     for n, (video, metadata) in enumerate(cfg['image_sets'].items()):
    #         print(video,metadata)
    #         label.draw_image_labels(project / metadata['img_path'] / 'multijoint.csv', cfg['joints'],
    #                                 cmap_name=cfg['cmap'] if 'cmap' in cfg else None)
    #     util.update_yaml(project / 'status.yaml', {'TrainingLabelsDrawn': True})
    #     _echo('Labels drawn. Check labeled images in image set directories')
    #     return
    # else:
    #     _echo('Drawing Labels OK')

    # # Join csv files of all image sets
    # # -------------------------------------------------------------------------------------------
    # if 'LabelsCollected' not in project_status or not project_status['LabelsCollected']:
    #     _echo('Preparing combined label file...')
    #     label.combine_labels(project / 'data')
    #     util.update_yaml(project / 'status.yaml', {'LabelsCollected': True})
    # else:
    #     _echo('Label Collection OK')

    # # Create shuffles
    # # -------------------------------------------------------------------------------------------
    # if 'Shuffled' not in project_status or not project_status['Shuffled']:
    #     _echo('Shuffling and splitting training set')
    #     for n in trange(cfg['num_shuffles'], leave=False):
    #         num_frames = int(cfg['num_frames'])
    #         f_train = float(cfg['train_fraction'])
    #         num_train = int(num_frames * f_train)
    #         num_testing = num_frames - num_train

    #         shuffle_name = 'shuffle{:03d}_{:.0f}pct-{}'.format(n, 100 * f_train, cfg['experimenter'])
    #         tqdm.write(shuffle_name)

    #         labels_csv = project / 'data' / 'joint_labels.csv'
    #         shuffle_path = project / 'shuffles' / shuffle_name

    #         # Create shuffle, training and test directories
    #         _ = [shuffle_path.joinpath(d).mkdir(exist_ok=True) for d in ['', 'train', 'test']]

    #         # Shuffle all images from the labeled example set with the specified fraction of training
    #         # to test images. The result is a list of labels for chosen images in a .mat or .csv file
    #         # to be used during training
    #         shuffle.shuffle(csv_file=labels_csv, train_fraction=f_train, destination=shuffle_path, joints=cfg['joints'],
    #                         boundary=cfg['boundary'])

    #         tqdm.write('Shuffling #{} w/ {}:{} train and test images'.format(n, num_train, num_testing))

    #     util.update_yaml(project / 'status.yaml', {'Shuffled': True})
    # else:
    #     _echo('Shuffling OK')

    # # Create directories and configuration files for pose-tensorflow (DeeperCut)
    # # -------------------------------------------------------------------------------------------
    # if 'DeepCutReady' not in project_status or not project_status['DeepCutReady']:
    #     _echo('Training preparation...')
    #     # Create training and testing directories for each shuffle
    #     shuffles = [d.resolve() for d in project.joinpath('shuffles').glob('*') if d.is_dir()]
    #     if not len(shuffles):
    #         util.update_yaml(project / 'status.yaml', {'Shuffled': False})
    #         print('No training sets found. Rerun wizard!')


    #     for shuffle_path in shuffles:
    #         _echo(str(shuffle_path))
    #         # Create training yaml
    #         train_set_path = shuffle_path / 'training.mat'
    #         if not train_set_path.resolve().exists():
    #             raise FileNotFoundError('Could not find "training.mat" file for shuffle "{}"'.format(shuffle_path))
    #         joints = cfg['joints']
    #         items2change = {'dataset': '../training.mat',
    #                         "num_joints": len(joints),
    #                         "all_joints": [[i] for i in range(len(joints))],
    #                         "all_joints_names": joints}

    #         resource_package = __name__  # Could be any module/package name
    #         resource_path = '/'.join(('..', 'resources', 'templates', 'training_pose_cfg.yaml'))

    #         pose_cfg_template = pkg_resources.resource_filename(resource_package, resource_path)

    #         # Create configuration yaml for training, and keep configuration data for the test configuration
    #         trainingdata = shuffle.training_pose_yaml(pose_cfg_template, items2change,
    #                                                   shuffle_path / 'train' / 'pose_cfg.yaml')

    #         # Keys to keep for the test configuration yaml
    #         keys2save = ['dataset', 'num_joints', 'all_joints', 'all_joints_names', 'net_type', 'init_weights',
    #                      'global_scale', 'location_refinement', 'locref_stdev']

    #         shuffle.test_pose_yaml(trainingdata, keys2save, shuffle_path / 'test/pose_cfg.yaml')
    #     util.update_yaml(project / 'status.yaml', {'DeepCutReady': True})
    # else:
    #     _echo('Training prep OK')

    # Training
    # -------------------------------------------------------------------------------------------
    if 'Trained' not in project_status or not project_status['Trained']:
        _echo('Start training...')
        shuffles = [d.resolve() for d in project.joinpath('shuffles').glob('*') if d.is_dir()]
        for shfl in shuffles:
            cfg_yaml_path = shfl / 'train/pose_cfg.yaml'
            _echo('Training starting for: {}'.format(cfg_yaml_path))
            training.train(cfg_yaml_path)
        return
    else:
        _echo('Training completed (to some degree')

    # Evaluation
    if 'Evaluated' not in project_status or not project_status['Evaluated']:
        _echo('Evaluation trained models')
        return
    else:
        _echo('Eavluation complete')

    # Inference
    if 'ReadyForUse' not in project_status or not project_status['ReadForUse']:
        _echo('What is left to do?')
        return
    else:
        _echo('Use me!')
        


