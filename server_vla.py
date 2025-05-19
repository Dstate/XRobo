import os
import torch

from models import create_model
from datasets import create_engine
from utils import RoboModelWrapper
import json

def main(path, ckpt_name, strict_load=True, host="0.0.0.0", port=8000):
    json_file = os.path.join(path, 'config.json')
    config = json.load(open(json_file, 'r'))
    
    model = create_model(**config)
    ckpt_file = os.path.join(path, f"{ckpt_name}.pth")
    model.load_state_dict(torch.load(ckpt_file, map_location='cpu'), strict=strict_load)
    model = RoboModelWrapper(model)
    _, agent = create_engine(**config)
    agent.set_policy(model)
    agent.run(path, ckpt_name, host=host, port=port)

if __name__ == '__main__':
    device_id = 7
    port = 8989 + device_id
    path = "runnings/bc_ddpm-libero_10"
    ckpt_name = "Model_ckpt_200000"
    torch.cuda.set_device(device_id)
    main(path, ckpt_name, False, "0.0.0.0", port)
