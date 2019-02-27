#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch10/bin/python

import os
import sys
import logging
import argparse
import yaml

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from inferno.io.transform.base import Compose
from inferno.extensions.criteria import SorensenDiceLoss
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

import neurofire.models as models
from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.loss_transforms import ApplyAndRemoveMask
from neurofire.criteria.loss_transforms import RemoveSegmentationFromTarget
from neurofire.criteria.loss_transforms import InvertTarget
from neurofire.datasets.isbi2012.loaders import get_isbi_loader_3d


# TODO implement with public MWS
# from neurofire.metrics.arand import ArandErrorFromMWS
from mws_metrics import ArandErrorFromMWS


logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_up_training(project_directory,
                    config,
                    data_config):

    # Get model
    model_name = config.get('model_name')
    model = getattr(models, model_name)(**config.get('model_kwargs'))

    criterion = SorensenDiceLoss()
    loss_train = LossWrapper(criterion=criterion,
                             transforms=InvertTarget())
    loss_val = LossWrapper(criterion=criterion,
                           transforms=Compose(RemoveSegmentationFromTarget(),
                                              InvertTarget()))

    # Build trainer and validation metric
    logger.info("Building trainer.")
    smoothness = 0.75

    offsets = data_config['volume_config']['segmentation']['affinity_config']['offsets']
    strides = [1, 10, 10]
    metric = ArandErrorFromMWS(average_slices=False, offsets=offsets,
                               strides=strides, randomize_strides=False)

    trainer = Trainer(model)\
        .save_every((1000, 'iterations'),
                    to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss_train)\
        .build_validation_criterion(loss_val)\
        .build_optimizer(**config.get('training_optimizer_kwargs'))\
        .evaluate_metric_every('never')\
        .validate_every((100, 'iterations'), for_num_iterations=1)\
        .register_callback(SaveAtBestValidationScore(smoothness=smoothness,
                                                     verbose=True))\
        .build_metric(metric)\
        .register_callback(AutoLR(factor=0.99,
                                  patience='100 iterations',
                                  monitor_while='validating',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous'))\

    logger.info("Building logger.")
    # Build logger
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every=(100, 'iterations'),
                                    log_histograms_every='never').observe_states(
        ['validation_input', 'validation_prediction, validation_target'],
        observe_while='validating'
    )

    trainer.build_logger(tensorboard,
                         log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


def load_checkpoint(project_directory):
    logger.info("Loading trainer from directory %s" % project_directory)
    trainer = Trainer().load(from_directory=project_directory,
                             filename='Weights/checkpoint.pytorch')
    return trainer


def training(project_directory,
             train_configuration_file,
             data_configuration_file,
             validation_configuration_file,
             max_training_iters=int(1e5),
             from_checkpoint=False):

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_isbi_loader_3d(data_configuration_file)
    data_config = yaml2dict(data_configuration_file)

    logger.info("Loading validation data loader from %s." % validation_configuration_file)
    validation_loader = get_isbi_loader_3d(validation_configuration_file)

    if from_checkpoint:
        trainer = load_checkpoint(project_directory)
    else:
        trainer = set_up_training(project_directory,
                                  config,
                                  data_config)
    trainer.set_max_num_iterations(max_training_iters)

    # Bind loader
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train',
                        train_loader).bind_loader('validate',
                                                  validation_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))

    # Go!
    logger.info("Lift off!")
    trainer.fit()


# configuration for the network
def make_train_config(train_config_file, affinity_config, gpus):
    template = './template_config/train_config_unet.yml'
    n_out = len(affinity_config['offsets'])
    template = yaml2dict(template)
    template['model_kwargs']['out_channels'] = n_out
    template['devices'] = gpus
    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


# configuration for training data
def make_data_config(data_config_file, affinity_config, n_batches):
    template = yaml2dict('./template_config/data_config.yml')
    template['volume_config']['segmentation']['affinity_config'] = affinity_config
    template['loader_config']['batch_size'] = n_batches
    template['loader_config']['num_workers'] = 6 * n_batches
    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


# configuration for validation data
def make_validation_config(validation_config_file, affinity_config):
    template = yaml2dict('./template_config/validation_config.yml')
    if affinity_config is not None:
        affinity_config.update({'retain_segmentation': True, 'add_singleton_channel_dimension': True})
    template['volume_config']['segmentation']['affinity_config'] = affinity_config
    with open(validation_config_file, 'w') as f:
        yaml.dump(template, f)


def get_mws_offsets():
    # direct 3d nhood for attractive edges
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
            # indirect 3d nhood for mutex edges
            [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
            # long range direct hood
            [0, -9, 0], [0, 0, -9],
            # inplane diagonal mutex edges
            [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
            # additional long range mutex edges
            [0, -27, 0], [0, 0, -27]]


def get_default_offsets():
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [0, -3, 0], [0, 0, -3],
            [0, -9, 0], [0, 0, -9],
            [0, -27, 0], [0, 0, -27]]


def copy_train_file(project_directory):
    from shutil import copyfile
    file_path = os.path.abspath(__file__)
    dst = os.path.join(project_directory, 'train.py')
    copyfile(file_path, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))
    parser.add_argument('--from_checkpoint', type=int, default=0)

    args = parser.parse_args()
    project_directory = args.project_directory
    from_checkpoint = bool(args.from_checkpoint)
    os.makedirs(project_directory, exist_ok=True)

    affinity_config = {'add_singleton_channel_dimension': True} #, 'ignore_label': None}
    offsets = get_mws_offsets()
    # offsets = get_default_offsets()
    affinity_config['offsets'] = offsets

    gpus = list(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    train_config = os.path.join(project_directory, 'train_config.yml')
    data_config = os.path.join(project_directory, 'data_config.yml')
    validation_config = os.path.join(project_directory, 'validation_config.yml')

    # only copy if files if we DON'T load from checkponit
    if not from_checkpoint:
        make_train_config(train_config, affinity_config, gpus)
        make_data_config(data_config, affinity_config, len(gpus))
        make_validation_config(validation_config, affinity_config)
        copy_train_file(project_directory)

    training(project_directory,
             train_config,
             data_config,
             validation_config,
             max_training_iters=args.max_train_iters,
             from_checkpoint=from_checkpoint)


if __name__ == '__main__':
    main()
