import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from .processors.BaseProcessor import build_base_processor
import h5py
import cv2
import torch
from typing import Any, Dict, Union
import traceback
import uvicorn
import json_numpy
from fastapi import FastAPI
from fastapi.responses import JSONResponse


class CalvinDataset(Dataset):
    def __init__(self, processor, chunk_length=6):
        self.processor = processor
        self.chunk_length = chunk_length
        self._load_metas()
    
    def _load_metas(self):
        dataset_statistics = self.processor.dataset_statistics
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
            np_action = f['rel_action'][()][cur_idx : cur_idx + self.chunk_length]
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

class CalvinAgent(object):
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
            agent_view_images = json_numpy.loads(payload["third_image"])
            wrist_view_images = json_numpy.loads(payload["wrist_image"])
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


def build_calvin_dataloader(processor, chunk_length=6,
                        batch_size=2, num_workers=2, shuffle=True, pin_mem=True, drop_last=True, 
                        world_size=1, global_rank=0):
    
    train_dataset = CalvinDataset(processor=processor, chunk_length=chunk_length)
    sampler = DistributedSampler(train_dataset, shuffle=shuffle, num_replicas=world_size, rank=global_rank) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=sampler, pin_memory=pin_mem, drop_last=drop_last)
    return train_dataloader

def build_calvin_agent(processor, use_ac=True):
    agent = CalvinAgent(processor, use_ac)
    return agent

def build_calvin_engine(dataset_path, processor_type='base', img_size=224, # processor
                        chunk_length=6, batch_size=2, num_workers=2, # dataloader
                        shuffle=True, pin_mem=True, drop_last=True, # dataloader
                        world_size=1, global_rank=0, # dataloader
                        use_ac=True, # agent
                        **kwargs):
    
    processor = build_base_processor(dataset_path, processor_type=processor_type, img_size=img_size, training=True)
    train_dataloader = build_calvin_dataloader(processor=processor, chunk_length=chunk_length,
                                               batch_size=batch_size, num_workers=num_workers, 
                                               shuffle=shuffle, pin_mem=pin_mem, drop_last=drop_last, 
                                               world_size=world_size, global_rank=global_rank)
    processor = build_base_processor(dataset_path, processor_type=processor_type, img_size=img_size, training=False)
    agent = build_calvin_agent(processor=processor, use_ac=use_ac)
    return train_dataloader, agent
