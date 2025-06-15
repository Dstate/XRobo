from .base_config import *

logger_type = "tensorboard"
# data engine
engine_name = "build_libero_engine"
dataset_path = "assets/data/libero"
processor_type = 'base'
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