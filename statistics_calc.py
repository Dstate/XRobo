import os
import os.path as osp
from pathlib import Path
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from utils import check_hdf5_structure

def get_args_parser():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--dataset_root', type=str)
    cfg = parser.parse_args()
    return cfg

def check_hdf5(dataset_path):
    hdf5_files = [str(file.resolve()) for file in Path(dataset_path).rglob('*.hdf5')]
    check_hdf5_structure(hdf5_files[0])

def build_dataset_statistics(dataset_path, cache_json_name='cache.json', action_key='action', proprio_key='proprio'):
    cache_json = osp.join(dataset_path, cache_json_name)
    if osp.isfile(cache_json):
        print('dataset statistics exits')
        dataset_statistics = json.load(open(cache_json, 'r'))
    else :
        print('Beginning to build dataset statistics...')
        hdf5_files = [str(file.resolve()) for file in Path(dataset_path).rglob('*.hdf5')]
        views = ['images0']
        traj_lens = []
        proprios = []
        actions = []
        # check all data
        for file in tqdm(hdf5_files):
            with h5py.File(file, 'r') as f:
                views = list(f['observation'].keys())
                traj_actions = f[action_key][()].astype('float32')
                traj_proprios = f[proprio_key][()].astype('float32')
                actions.append(traj_actions)
                proprios.append(traj_proprios)
                traj_lens.append(traj_actions.shape[0])
        # calculate statistics
        actions = np.concatenate(actions, axis=0)
        proprios = np.concatenate(proprios, axis=0)
        action_max = actions.max(axis=0).tolist()
        action_min = actions.min(axis=0).tolist()
        proprio_max = proprios.max(axis=0).tolist()
        proprio_min = proprios.min(axis=0).tolist()
        dataset_statistics = dict(views=views, action_max=action_max, action_min=action_min,
                                  proprio_max = proprio_max, proprio_min = proprio_min,
                                  traj_paths=hdf5_files, traj_lens=traj_lens, 
                                  action_key=action_key, proprio_key=proprio_key)
        with open(cache_json, 'w') as f:
            json.dump(dataset_statistics, f, indent=4)
    return dataset_statistics


if __name__ == '__main__':
    config = get_args_parser()
    check_hdf5(config.dataset_root)

    if 'libero' in config.dataset_root:
        print('processing libero data...')
        build_dataset_statistics(config.dataset_root)
        print('ok')
    if 'VLABench' in config.dataset_root:
        print('processing VLABench data...')
        build_dataset_statistics(config.dataset_root)
        print('ok')
    elif 'calvin' in config.dataset_root:
        print('processing calvin data...')
        build_dataset_statistics(config.dataset_root, action_key='rel_action')
        print('ok')
    elif 'simpler' in config.dataset_root:
        if 'bridge' in config.dataset_root:
            print('processing simpler_bridge data...')
            build_dataset_statistics(config.dataset_root)
            print('ok')
        elif 'RT1' in config.dataset_root:
            print('processing simpler_bridge data...')
            build_dataset_statistics(config.dataset_root, proprio_key='action')
            print('ok')
