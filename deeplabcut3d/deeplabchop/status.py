from pathlib import Path
from deeplabchop.util import read_yaml


def read_status(path):
    path = Path(path)
    yaml_path = path / 'status.yaml' if path.is_dir() else path

    return read_yaml(yaml_path)


def show_status(path):
    yd = read_status(path)
    print('Status of {}:'.format(path))

    for k, i in yd.items():
        print('\t{}: {}'.format(k, i))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    cli_args = parser.parse_args()

    show_status(cli_args.path)
