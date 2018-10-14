import os
from pathlib import Path

from pose_tensorflow import train as ptf_train


def train(config):
    """Starts training with pose-tensorflow.

    Additionally keeps track of working directory, such that when exiting
    it will return to working directory it was launched from.

    This works around pose-tensorflow expecting to be in the model location.
    """
    start_path = Path.cwd()
    config_path = Path(config).resolve()
    wd = config_path.parent
    os.chdir(str(wd))
    try:
        ptf_train.train(config_path)
    except BaseException as e:
        raise e
    finally:
        os.chdir(str(start_path))
