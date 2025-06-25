import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from .processors.BaseProcessor import build_base_processor
from .LiberoEngine import LiberoDataset, LiberoAgent
from .CalvinEngine import CalvinDataset, CalvinAgent
from .SimplerBridgeEngine import SimplerBridgeDataset, SimplerBridgeAgent
from .SimplerRT1Engine import SimplerRT1Dataset, SimplerRT1Agent

name2dataset = {
    'libero': LiberoDataset,
    'calvin': CalvinDataset,
    'simpler_bridge': SimplerBridgeDataset,
    'simpler_rt1': SimplerRT1Dataset
}

name2agent = {
    'libero': LiberoAgent,
    'calvin': CalvinAgent,
    'simpler_bridge': SimplerBridgeAgent,
    'simpler_rt1': SimplerRT1Agent
}

name2processor = {
    'libero': 'base',
    'calvin': 'base_color',
    'simpler_bridge': 'base_color',
    'simpler_rt1': 'base_color'
}

def collate_fn(batch):
    for b in batch:
        b['cur_proprios'] = torch.zeros(9)
        if b['cur_images'].shape[0] == 1:
            b['cur_images'] = torch.cat([b['cur_images'], torch.zeros_like(b['cur_images'])], dim=0)
    return default_collate(batch)

def build_joint_dataloader(dataset_name_list=['libero', 'calvin', 'simpler_bridge', 'simpler_rt1'],  img_size=224, # processor
                           dataset_path_list = ['assets/data/libero', 'assets/data/calvin', 'assets/data/simpler_bridge', 'assets/data/simpler_rt1'], # processor
                           chunk_length=6, batch_size=2, num_workers=2, # dataloader
                           shuffle=True, pin_mem=True, drop_last=True, # dataloader
                           world_size=1, global_rank=0, # dataloader
                           use_ac=True, # agent
                           **kwargs):
    
    assert len(dataset_name_list) == len(dataset_path_list)

    dataset_list = []
    agent_dict = {}
    for name, path in zip(dataset_name_list, dataset_path_list):
        processor = build_base_processor(path, processor_type=name2processor[name], img_size=img_size, training=True)
        dataset = name2dataset[name](processor, chunk_length=chunk_length)
        dataset_list.append(dataset)
        processor = build_base_processor(path, processor_type=name2processor[name], img_size=img_size, training=False)
        agent = name2agent[name](processor=processor, use_ac=use_ac)
        agent_dict[name] = agent

    joint_dataset = ConcatDataset(dataset_list)
    sampler = DistributedSampler(joint_dataset, shuffle=shuffle, num_replicas=world_size, rank=global_rank) 
    train_dataloader = DataLoader(joint_dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=sampler, pin_memory=pin_mem, drop_last=drop_last, collate_fn=collate_fn)
    return train_dataloader, agent_dict
