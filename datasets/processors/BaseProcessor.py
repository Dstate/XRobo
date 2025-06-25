import os
import os.path as osp
import torch
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py
import cv2
import torch
from typing import Any, Dict, Union
import traceback
import uvicorn
import json_numpy
from fastapi import FastAPI
from fastapi.responses import JSONResponse

def build_dataset_statistics(dataset_path, cache_json_name='cache.json'):
    cache_json = osp.join(dataset_path, cache_json_name)
    assert osp.isfile(cache_json), 'dataset statistics don\'t exit'
    dataset_statistics = json.load(open(cache_json, 'r'))
    return dataset_statistics

def build_base_transform(n_px, aug=True, to_tensor=True, apply_norm=True,
                        crop_scale=(0.75,1.0), crop_ratio=(0.75, 1.33), crop_prob=1.0, flip_prob=0.5, jitter_prob=0.5, 
                        jitter_bright=0.1, jitter_contrast=0, jitter_saturation=0, jitter_hue=0,
                        norm_mean = (0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):

    base_transform = []
    # augmentation and resize
    if aug:
        base_transform.append(A.RandomResizedCrop(size=(n_px,n_px), p=crop_prob,
                                                  scale=crop_scale, ratio=crop_ratio))
    else :
        base_transform.append(A.Resize(height=n_px, width=n_px))
    # normalization
    if apply_norm:
        base_transform.append(A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0, p=1.0))
    # convert to tensor
    if to_tensor:
        base_transform.append(ToTensorV2())
    # build transform
    base_transform = A.ReplayCompose(base_transform)
    return base_transform

class BaseProcessor(object):
    def __init__(self, dataset_path, img_transform):
        self.img_transform = img_transform
        dataset_statistics = build_dataset_statistics(dataset_path)
        self.dataset_statistics = dataset_statistics
        self.action_max = np.array(dataset_statistics['action_max'])
        self.action_min = np.array(dataset_statistics['action_min'])
        self.proprio_max = np.array(dataset_statistics['proprio_max'])
        self.proprio_min = np.array(dataset_statistics['proprio_min'])
        # fix parameters
        self.action_length = 7
        self.proprio_length = 9
    
    def preprocess_image(self, img, replay_params=None):
        if replay_params == None:
            transformed = self.img_transform(image=img)
            transformed_image = transformed['image']
            replay_params = transformed['replay']
        else :
            transformed = A.ReplayCompose.replay(replay_params, image=img)
            transformed_image = transformed['image']
        return transformed_image, replay_params
    
    def preprocess_action(self, action):
        action = (action - self.action_min) / (self.action_max - self.action_min) * 2 - 1
        action = torch.flatten(torch.from_numpy(action))
        return action
    
    def preprocess_proprio(self, proprio):
        proprio = (proprio - self.proprio_min) / (self.proprio_max - self.proprio_min) * 2 - 1
        proprio = torch.flatten(torch.from_numpy(proprio))
        return proprio
    
    def postprocess_action(self, tensor_flatten_action):
        # action B 42 -> B 6 7
        B, _ = tensor_flatten_action.shape
        action = tensor_flatten_action.reshape(B, -1, self.action_length)
        action[..., -1] = torch.sign(action[..., -1])
        action = (action + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        return action.numpy()

def build_base_processor(dataset_path, processor_type, img_size=224, training=True):
    if processor_type == 'base':
        img_transform = build_base_transform(n_px=img_size, aug=training)
        processor = BaseProcessor(dataset_path, img_transform)
        return processor
    elif processor_type == 'base_clip':
        img_transform = build_base_transform(n_px=img_size, aug=training)
        processor = BaseProcessor(dataset_path, img_transform,
                        norm_mean = (0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711))
        return processor
    else:
        raise NotImplementedError
