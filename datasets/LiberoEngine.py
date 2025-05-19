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

def build_base_transform(n_px, aug=True, to_tensor=True, apply_norm=True,
                        crop_scale=(0.75,1.0), crop_ratio=(0.75, 1.33), crop_prob=1.0, flip_prob=0.5, jitter_prob=0.5, 
                        jitter_bright=0.1, jitter_contrast=0, jitter_saturation=0, jitter_hue=0,
                        norm_mean = (0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
                        # norm_mean = (0.48145466, 0.4578275, 0.40821073), norm_std=(0.26862954, 0.26130258, 0.27577711)):

    base_transform = []
    # augmentation and resize
    if aug:
        base_transform.append(A.RandomResizedCrop(size=(n_px,n_px), p=crop_prob,
                                                  scale=crop_scale, ratio=crop_ratio))
        # base_transform.append(A.VerticalFlip(p=flip_prob))
        # base_transform.append(A.HorizontalFlip(p=flip_prob))
        # base_transform.append(A.ColorJitter(brightness=jitter_bright, contrast=jitter_contrast, 
        #                                     saturation=jitter_saturation, hue=jitter_hue, p=jitter_prob))
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

def build_dataset_statistics(dataset_path, cache_json_name='cache.json'):
    cache_json = osp.join(dataset_path, cache_json_name)
    assert osp.isfile(cache_json), 'dataset statistics don\'t exit'
    dataset_statistics = json.load(open(cache_json, 'r'))
    return dataset_statistics

class LiberoProcessor(object):
    def __init__(self, dataset_path, img_size=224, training=True):
        self.img_transform = build_base_transform(n_px=img_size, aug=training)
        dataset_statistics = build_dataset_statistics(dataset_path)
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


class LiberoDataset(Dataset):
    def __init__(self, dataset_path, processor, chunk_length=6):
        self.processor = processor
        self.dataset_path = dataset_path
        self.chunk_length = chunk_length
        self._load_metas()
    
    def _load_metas(self):
        dataset_statistics = build_dataset_statistics(self.dataset_path)
        traj_paths = dataset_statistics['traj_paths']
        traj_lens = dataset_statistics['traj_lens']
        self.views = dataset_statistics['views']
        self.main_view = self.views[0]
        self.metas = []
        for i in range(len(traj_paths)):
            self.metas.extend([(traj_paths[i], j, traj_lens[i]) for j in range(traj_lens[i])])

    def _load_from_raw_traj(self, traj_path, cur_idx, goal_idx):
        with h5py.File(traj_path, 'r') as f:
            # load images from all views
            raw_images = []
            for view in self.views:
                raw_img = cv2.imdecode(f['observation'][view][cur_idx], cv2.IMREAD_COLOR)
                raw_images.append(raw_img)
            # load goal images from all views
            goal_images = []
            for view in self.views:
                raw_img = cv2.imdecode(f['observation'][view][goal_idx], cv2.IMREAD_COLOR)
                goal_images.append(raw_img)
            # load actions with chunking
            np_action = f['action'][()][cur_idx : cur_idx + self.chunk_length]
            if len(np_action) < self.chunk_length:
                cnt = self.chunk_length - len(np_action)
                padding = np.array([[0., 0., 0., 0., 0., 0., np_action[-1][-1]]]).repeat(cnt, axis=0)
                np_action = np.concatenate([np_action, padding], axis=0)
            # load proprio
            raw_proprio = f['proprio'][()][cur_idx]
            # load instruction
            instruction = f['language_instruction'][()].decode('utf-8')
        return raw_images, goal_images, np_action, raw_proprio, instruction

    def __len__(self):
        return len(self.metas) 
    
    def __getitem__(self, index):
        meta = self.metas[index]
        raw_images, goal_images, np_action, raw_proprio, instruction = self._load_from_raw_traj(meta[0], meta[1], meta[2]-1)
        final_images = torch.stack([self.processor.preprocess_image(img)[0] for img in raw_images])
        goal_images = torch.stack([self.processor.preprocess_image(img)[0] for img in goal_images])
        final_action = self.processor.preprocess_action(np_action) # 42
        final_proprio = self.processor.preprocess_proprio(raw_proprio)
        item = {
            'cur_images': final_images,
            'goal_images': goal_images,
            'cur_actions': final_action,
            'cur_proprios': final_proprio,
            'instruction': instruction,
            'traj_path': meta[0],
            'cur_idx': meta[1],
        }
        return item

class LiberoAgent(object):
    def __init__(self, processor, use_ac = True):
        super().__init__()
        self.use_ac = use_ac
        self.constant = 10000
        self.processor = processor
        self.policy = None

    def set_policy(self, policy):
        assert hasattr(policy, 'generate') and callable(getattr(policy, 'generate')), \
        "The policy must have a callable 'generate' method."
        self.policy = policy

    def _init_action_chunking(self, eval_horizon: int=600, num_samples: int=1):
        self.all_time_actions = np.ones([num_samples, eval_horizon, eval_horizon+50, 7]) * self.constant
    
    def get_ac_action(self, actions, t: int, k: float=0.25):
        B, N, D = actions.shape
        self.all_time_actions[:, [t], t:t+N] = np.expand_dims(actions, axis=1)   # B, horizon, horizon+ac_num, 7
        actions_for_curr_step = self.all_time_actions[:, :, t]  # B, horizon, 7
        actions_populated = np.all(actions_for_curr_step != self.constant, axis=-1)  # B, horizon
        actions_for_curr_step = actions_for_curr_step[actions_populated].reshape(B, -1, D)  # B, N, 7
        exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[1]))  # N, 1
        exp_weights = (exp_weights / exp_weights.sum()).reshape(1, -1, 1)
        actions = (actions_for_curr_step * exp_weights).sum(axis=1)
        actions[..., -1] = np.sign(actions[..., -1])
        return actions
    
    def get_action(self, agent_view_images, wrist_view_images, raw_proprio, instruction, t=-1):
        # agent_view_images B H W 3
        # wrist_view_images B H W 3
        # raw_proprio B 9
        # instruction ['xxx', ..., 'xxx']

        agent_view_images = torch.stack([self.processor.preprocess_image(image)[0] for image in agent_view_images]).unsqueeze(1)
        wrist_view_images = torch.stack([self.processor.preprocess_image(image)[0] for image in wrist_view_images]).unsqueeze(1)
        final_images = torch.cat([agent_view_images, wrist_view_images], dim=1)
        final_proprio = torch.stack([self.processor.preprocess_proprio(proprio) for proprio in raw_proprio])
        batch = {
            'cur_images': final_images,
            'cur_proprios': final_proprio,
            'instruction': instruction,
        }
        actions, _ = self.policy.generate(**batch)
        actions = self.processor.postprocess_action(actions)
        if self.use_ac:
            assert t >= 0, f"Invalid value for t: {t}. In action chunking, t must be equal to current rollout step."
            smoothed_actions = self.get_ac_action(actions, t)
            # smoothed_actions[:, -1] = actions[:, 0, -1]
            actions = smoothed_actions
        else :
            actions = actions[:, 0, :]
        return actions

    def infer(self, payload: Dict[str, Any]):
        # agent_view_images B H W 3
        # wrist_view_images B H W 3
        # raw_proprio B 9
        # instruction ['xxx', ..., 'xxx']
        # eval_horizon 600
        # t 0
        print('recieve a request')
        try:    
            agent_view_images = json_numpy.loads(payload["agent_view_images"])
            wrist_view_images = json_numpy.loads(payload["wrist_view_images"])
            raw_proprio = json_numpy.loads(payload["proprio"])
            instruction, eval_horizon, t = payload['instruction'], payload['eval_horizon'], payload['t']

            if t == 0:
                self._init_action_chunking(eval_horizon, agent_view_images.shape[0])

            pred_action = self.get_action(agent_view_images, wrist_view_images, raw_proprio, instruction, t)
            return JSONResponse(content={
                "pred_action": pred_action.tolist()
            })
        except:
            error_str = traceback.format_exc()
            print("Error occurred:", error_str)
            return JSONResponse(content={
                "pred_action": None,
                "error_str": error_str
            })

    def info_query(self):
        return JSONResponse(content={
            "save_path": self.save_path
        })

    def run(self, output_dir, ckpt_name, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.save_path = osp.join(output_dir, f"Eval_{ckpt_name}")
        self.app.post("/act")(self.infer)
        self.app.post("/info")(self.info_query)
        uvicorn.run(self.app, host=host, port=port)

def build_libero_processor(dataset_path, img_size=224, training=True):
    processor = LiberoProcessor(dataset_path=dataset_path, img_size=img_size, training=training)
    return processor

def build_libero_dataloader(dataset_path, processor, chunk_length=6,
                        batch_size=2, num_workers=2, shuffle=True, pin_mem=True, drop_last=True, 
                        world_size=1, global_rank=0):
    
    train_dataset = LiberoDataset(dataset_path=dataset_path, processor=processor, chunk_length=chunk_length)
    sampler = DistributedSampler(train_dataset, shuffle=shuffle, num_replicas=world_size, rank=global_rank) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=sampler, pin_memory=pin_mem, drop_last=drop_last)
    return train_dataloader

def build_libero_agent(processor, use_ac=True):
    agent = LiberoAgent(processor, use_ac)
    return agent

def build_libero_engine(dataset_path, img_size=224, # processor
                        chunk_length=6, batch_size=2, num_workers=2, # dataloader
                        shuffle=True, pin_mem=True, drop_last=True, # dataloader
                        world_size=1, global_rank=0, # dataloader
                        use_ac=True, # agent
                        **kwargs):
    
    processor = build_libero_processor(dataset_path, img_size=img_size, training=True)
    train_dataloader = build_libero_dataloader(dataset_path, processor=processor, chunk_length=chunk_length,
                                               batch_size=batch_size, num_workers=num_workers, 
                                               shuffle=shuffle, pin_mem=pin_mem, drop_last=drop_last, 
                                               world_size=world_size, global_rank=global_rank)
    processor = build_libero_processor(dataset_path, img_size=img_size, training=False)
    agent = build_libero_agent(processor=processor, use_ac=use_ac)
    return train_dataloader, agent
