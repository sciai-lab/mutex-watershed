#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/inferno/bin/python

import os
import argparse
import h5py

from inferno.trainers.basic import Trainer
from inferno.utils.io_utils import yaml2dict
from neurofire.inference import SimpleInferenceEngine
from neurofire.datasets.isbi2012.loaders.raw import RawVolumeHDF5


def load_volume(inference_config):
    config = yaml2dict(inference_config)
    vol_config = config['volume_config']['raw']
    slicing_config = config['slicing_config']
    return RawVolumeHDF5(**vol_config, **slicing_config)


def run_inference(project_dir, out_file, inference_config):

    print("Loading model...")
    model = Trainer().load(from_directory=os.path.join(project_dir, "Weights"), best=True).model
    print("Loading dataset...")
    dataset = load_volume(inference_config)

    engine = SimpleInferenceEngine.from_config(inference_config, model)
    print("Run prediction...")
    out = engine.infer(dataset)
    if out_file != '':
        print("Save prediction to %s ..." % out_file)
        with h5py.File(out_file, 'w') as f:
            f.create_dataset('data', data=out, compression='gzip')
    return out


def set_device(device):
    print("Setting cuda devices to", device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('out_file', type=str, default='')
    parser.add_argument('--inference_config', type=str, default='template_config/inf_config.yml')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.device != 0:
        set_device(args.device)
    run_inference(args.project_dir, args.out_file, args.inference_config)


if __name__ == '__main__':
    main()
