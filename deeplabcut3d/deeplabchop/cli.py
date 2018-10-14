import os
import click
from pathlib import Path

import deeplabchop

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(invoke_without_command=True)
# @click.version_option()
@click.option('-v', '--verbose', is_flag=True, help='Verbose printing')
@click.pass_context
def main(ctx, verbose):
    if ctx.invoked_subcommand is None:
        click.echo('DeepLabChop v0.0.1')
        click.echo(main.get_help(ctx))


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument('project')
@click.argument('experimenter')
@click.argument('videos', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-d', '--wd', 'working_directory',
              type=click.Path(exists=True, file_okay=False, resolve_path=True), default=Path.cwd(),
              help='Directory to create project in. Default is cwd().')
@click.option('-c', '--copy_videos', 'working_directory',
              type=click.Path(exists=True, file_okay=False, resolve_path=True), default=Path.cwd(),
              help='Directory to create project in. Default is cwd().')
@click.pass_context
def new(_, *args, **kwargs):
    """Boilerplate creator for a new PROJECT by EXPERIMENTER.

    Can be provided with a list of VIDEOS. This is useful to include videos that will
    not be stored within the project hierarchy and only linked as relative paths
    in the config.yaml (e.g. avoid copies of large video files). Otherwise videos can be
    copied into the project 'videos' directory after the project structure was created
    and added with the `prepare` tool.

    Creates the PROJECT-EXPERIMENTER-DATE directory with proper naming, structure,
    and basic configuration file templates.
    """
    deeplabchop.new.project(*args, **kwargs)


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument('project')
@click.pass_context
def wizard(_, project):
    """Step through project creation steps """
    deeplabchop.wizard.step(project)


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument('config')
@click.pass_context
def train(_, config):
    """Train a model with training data."""
    deeplabchop.training.train(config)


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument('video', type=click.Path(exists=True, dir_okay=False))
@click.argument('destination', type=click.Path(exists=True, dir_okay=False))
@click.argument('num_frames', type=int)
@click.pass_context
def extract(_, video, destination, num_frames):
    """Manually trigger frame extraction from a video into a directory."""
    extract.extract_frames(video, destination, num_frames)


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument('config')
@click.argument('videos', nargs=-1)
@click.pass_context
def predict(_, config, videos):
    """Evaluate a model."""
    for video in videos:
        deeplabchop.predict.predict_video(config, video)


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument('results')
@click.argument('videos', nargs=-1)
@click.pass_context
def draw(_, results, videos):
    """Draw predictions into images or videos"""
    for video in videos:
        deeplabchop.draw.draw_predictions_video(results, video)


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument('cfg_path')
@click.pass_context
def label(_, cfg_path):
    """Training image annotation assistance.
    """
    click.echo('label assist', cfg_path)


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument('path')
@click.pass_context
def status(_, *args, **kwargs):
    """Show status information about a config file.
    Training progress, log file status, etc.

    Input:
        cfg_path: Path to config file
    """
    deeplabchop.status.show_status(*args, **kwargs)


if __name__ == '__main__':
    main()
