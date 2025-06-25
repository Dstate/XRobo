from datasets import create_engine
from utils import check_dict_structure
from models.encoders.DecisionNCE import DecisionNCE
from tqdm import tqdm 

train_dataloader, _ = create_engine('build_libero_engine', dataset_path = 'assets/data/libero', batch_size=128)

model = DecisionNCE()


for batch in tqdm(train_dataloader):
    check_dict_structure(batch)
    break
    # visual_input = batch['cur_images'][:,0,...]
    # goal_input = batch['goal_images'][:, 0, ...]
    # text_input = batch['instruction']
    # image_features, text_features, cos_sim = model(visual_input, goal_input, text_input)
    
    # print(image_features.shape, text_features.shape, cos_sim)
    