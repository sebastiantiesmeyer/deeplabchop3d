from pathlib import Path
import math

import pandas as pd
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from skimage import draw
from skvideo import io
import matplotlib.pyplot as plt

PCUTOFF = .1
DEFAULT_COLORMAP = 'RdYlGn'


def draw_predictions_video(results, video, markersize=4, cmap_name=DEFAULT_COLORMAP, p_cutoff=PCUTOFF):
    """Draw pre-calculated joint locations into video frames.
    """
    video_path = Path(video)

    clip = VideoFileClip(str(video_path)).to_RGB()

    ny, nx, fps = clip.h, clip.w, clip.fps

    df = pd.read_hdf(results)
    scorer = 'reichler'

    joints = list(np.unique(df.columns.get_level_values(1)))

    cmap = plt.cm.get_cmap(cmap_name, len(joints))

    clip_out = io.FFmpegWriter(str(video_path.with_suffix('.labeled.mp4')),
                               inputdict={'-r': str(clip.fps)},
                               outputdict={'-r': str(clip.fps)})

    num_frames = math.ceil(clip.duration*clip.fps)

    for index, frame in enumerate(tqdm(clip.iter_frames(dtype='uint8'), total=num_frames)):
        for bpindex, bp in enumerate(joints):
            if df[scorer][bp]['likelihood'].values[index] > p_cutoff:
                xc = int(df[scorer][bp]['x'].values[index])
                yc = int(df[scorer][bp]['y'].values[index])
                # rr, cc = circle_perimeter(yc, xc, radius)
                # tqdm.write(str((xc, yc, bp)))
                rr, cc = draw.circle(yc, xc, markersize, shape=(ny, nx))
                frame[rr, cc, :] = [c * 255 for c in cmap(bpindex)[:3]]

        clip_out.writeFrame(frame)

    clip_out.close()
    clip.close()


if __name__ == '__main__':
    pass
