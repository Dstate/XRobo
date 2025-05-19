import os
import time
import torch
import random
import argparse
import numpy as np
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import Logger, SmoothedValue
from utils import RoboModelWrapper, DataLoaderWithTimeWrapper
from utils import py2dict,merge_dict_to_dict, format_time_hms
from utils import init_ddp_env, close_ddp_env, save_checkpoint
from utils import CosineAnnealingWarmUpRestarts, check_dict_structure
from models import create_model
from datasets import create_engine

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--output_dir', type=str)
    cfg = parser.parse_args()
    config = py2dict(cfg.config_name)
    config = merge_dict_to_dict(vars(cfg), config)
    return config

def train(config, logger, model, train_loader, optimizer, lr_scheduler):
    start_time = time.time()
    loss_recod = SmoothedValue()

    for ep_iter, (batch, data_time) in enumerate(train_loader):
        iter_start_time = time.time()
        
        # calculate loss and update model
        loss, loss_metric = model(**batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loss_recod.update(loss)
        
        # log training message
        loss_metric.update({'iter': ep_iter,
                            'smoothed_loss': loss_recod.avg,
                            'lr': lr_scheduler.get_last_lr()[0]})
        logger.log_metric(loss_metric)
        if ep_iter % config['log_interval'] == 0 or ep_iter == len(train_loader)-1:
            current_time = time.time()
            elapsed_time_sec, elapsed_time = format_time_hms(current_time, start_time)
            iter_time_sec, iter_time = format_time_hms(current_time, iter_start_time)
            data_time_sec, data_time = format_time_hms(data_time)
            eta_sec, eta = format_time_hms(iter_time_sec * (len(train_loader)-ep_iter-1))
            logger.log_msg_every(
                f"Iter [{ep_iter}/{len(train_loader)}], "
                f"loss: {loss_metric['loss']:.4f}, "
                f"lr: {loss_metric['lr']:.6f}, "
                f"iter_time: {iter_time_sec:.4f}s, "
                f"data_time: {data_time_sec:.4f}s, "
                f"elapsed_time: {elapsed_time}, "
                f"eta: {eta}, ")
        
        # save checkpoint
        if (ep_iter + 1) % config['save_interval'] == 0 or (ep_iter + 1) == len(train_loader):
            logger.log_msg(f"Saving ckpt : [{ep_iter + 1}]")
            unwraped_model = model.module.unwrap()
            if config['global_rank'] == 0:
                save_checkpoint(ep_iter, unwraped_model, optimizer, lr_scheduler, 
                                config['output_dir'], f"ckpt_{ep_iter + 1}", True)
            dist.barrier()
            logger.log_msg(f"Save ok")

def prepare_training_components(config):
    # build model and engine
    model = create_model(**config)
    model = RoboModelWrapper(model)
    train_loader, agent = create_engine(**config)
    train_loader = DataLoaderWithTimeWrapper(train_loader, total_iters=config['num_iters'])
    
    # build optimizer and lr_scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_warmup=config['warm_steps'], 
                                    T_max=config['num_iters'], eta_min=config['eta_min_lr'])

    # setup ddp
    model = DistributedDataParallel(model)
    return model, train_loader, optimizer, lr_scheduler

def main_ddp(config):
    world_size, global_rank, local_rank = init_ddp_env()
    config['world_size'] = world_size
    config['global_rank'] = global_rank
    config['local_rank'] = local_rank
    
    seed_everything(config['seed'])
    logger = Logger(**config)
    logger.log_msg_every("init env ok")
    logger.log_msg_every(f"current device is: {torch.cuda.current_device()}")    
    logger.log_msg_every(f"current seed is: {config['seed']}")

    # training
    model, train_loader, optimizer, lr_scheduler = prepare_training_components(config)
    train(config, logger, model, train_loader, optimizer, lr_scheduler)
    close_ddp_env()

if __name__ == '__main__':
    config = get_args_parser()
    main_ddp(config)