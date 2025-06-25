from .base_config import *

logger_type = "tensorboard"
# data engine
engine_name = "build_joint_dataloader"
dataset_name_list=['libero', 'calvin', 'simpler_bridge', 'simpler_rt1']
dataset_path_list = ['assets/data/libero', 'assets/data/calvin', 'assets/data/simpler_bridge', 'assets/data/simpler_rt1']
shuffle=True
pin_mem=True
drop_last=True
use_ac=True
img_size=224
batch_size=32
num_workers=8
chunk_length=6

# model settings
model_name = "bc_policy_ddpm_res34_libero"