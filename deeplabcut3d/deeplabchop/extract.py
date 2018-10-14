import os
import math
import sys
from pathlib import Path

import numpy as np
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader

from skimage import io
from skimage.util import img_as_ubyte
from tqdm import tqdm

import imageio
import h5py

# Is there a way to check if that's needed?
imageio.plugins.ffmpeg.download()


def crop_frame(image, x1, y1, x2, y2):
    x2 = image.shape[1] - 1 if x2 < 0 else x2
    y2 = image.shape[0] - 1 if y2 < 0 else y2

    x_bad = any([0 > x >= image.shape[0] for x in [x1, x2]])
    y_bad = any([0 > y >= image.shape[1] for y in [y1, y2]])
    if x_bad or y_bad:
        raise ValueError('Crop bounding box exceeds image dimensions.')

    cropped = image[y1:y2, x1:x2]
    return cropped


def get_frame(video, t_frame=0.0):
    """Crop a single frame from the video to check cropping result. Stored alongside video.

    :param video: Path to video
    :param t_frame: position to extract example frame from, in seconds. Default: 0
    :return: image

    """
    clip = FFMPEG_VideoReader(str(video)) 
    image = clip.get_frame(t_frame)

    return image


def crop_examples(video, crop, destination=None):
    video_path = Path(video)
    dst_path = Path(video_path.parent if destination is None else destination).resolve()

    image_original = get_frame(video)
    image_cropped = crop_frame(image_original, *crop)

    io.imsave(dst_path / video_path.with_suffix('.original.png').name, image_original)
    io.imsave(dst_path / video_path.with_suffix('.cropped.png').name, image_cropped)


def extract_frames(video_data,video_root, destination, num_frames, seed=0):
    """Extracts [num_frames] images from a video and stores """
    
    np.random.seed(seed)

    file = h5py.File(destination,'w')
    X = None
    for n, (video, metadata) in enumerate(video_data.items()):
        crop = list(map(int, metadata['crop'].split(',')))

        if crop is not None:
            x1, y1, x2, y2 = crop
        else:
            x1,y1 = [0,0]
            x2,y2 = [-1,-1]

        video_path = video_root/Path(video)

           
        print('Extracting {} frames with seed {} from {}'.format(num_frames, seed, video_path.name))
        print(crop)

        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError('Video "{}" not found.'.format(video_path))

        # img_path = Path(destination).absolute()

        # TODO: Exception handling for faulty videos
        # Warning: Handles to VideoClip object are fickle, and getting them stuck is pretty easy.
        # Best always handle them with a context manager.
        clip =  FFMPEG_VideoReader(str(video_path))# as clip:
        print('Video duration: {} s, {} fps, uncropped frame size: {}'.format(clip.duration, clip.fps, clip.size))

        # Crop frames

        #     print('Cropping box {}->{}'.format((x1, y1), (x2, y2)))
        #     clip = clip.crop(x1, y1, x2, y2)

        num_frames_clip = int(clip.duration * clip.fps)
        #padding = int(math.ceil(math.log10(num_frames_clip)))  # file name padding

        # print('Storing images in {}'.format(img_path))

        # Grab frames from video and store as png file

        frame_indices = sorted(np.random.randint(0, max(1,num_frames_clip-11), num_frames))

        for idx in tqdm(np.diff([0]+frame_indices)):
            print(idx)
            image = []
            clip.skip_frames(idx)
            for i in range(10):
                frame = (clip.read_frame())#idx+i / clip.fps))
                # print(image)
                image.append(frame[y1:y2,x1:x2,:])
            
            if X is None:
                X = file.create_dataset('X',(1,10,y2-y1,x2-x1,3),
                            chunks=(1,10,y2-y1,x2-x1,3),
                            maxshape=(None,10,y2-y1,x2-x1,3),
                            compression="gzip", dtype = 'i')
            else:
                X.resize((X.shape[0]+1,)+X.shape[1:])
            
            X[-1,:,:,:,:]=image
            # print(X[-1,:,:,:,:])

    file.close()
                # fname = 'img{idx:0{pad}d}.png'.format(idx=idx, pad=padding)

                # try:
                #     io.imsave(img_path / fname, image)
                # except FileExistsError:
                #     print('{} exists already. Skipping.'.format(fname))




