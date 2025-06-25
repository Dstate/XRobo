
from timm.layers import Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from .encoders.DecisionNCE import DecisionNCE
from .components.MlpResNet import MlpResNet
from .components.ResNet import FilmResNet
from .components.ActionHead import BaseHead, DDPMHead

class BCPolicy(nn.Module):
    def __init__(
        self,
        latent_proj,
        proprio_input_dim = 9,
        proprio_hidden_dim = 32,
        vision_backbone_name: str = "resnet34",
        policy_num_blocks = 3,
        policy_hidden_dim = 256,
        action_size = 7,
        chunk_length = 4,
        num_views = 2,
        decoder_head = 'base',
        loss_func = nn.MSELoss,
        loss_func_conig = dict(reduction='mean')
    ):
        super().__init__()
        # condition encoder
        self.latent_proj = latent_proj
        self.latent_proj.requires_grad_(False)
        self.cond_dim = 1024

        # vision encoder
        self.num_views = num_views
        self.vision_encoder = FilmResNet(image_dim=3, cond_dim=self.cond_dim, backbone_name=vision_backbone_name)
        self.vision_dim = self.vision_encoder.vision_dim * self.num_views

        # proprio encoder
        self.proprio_dim = proprio_input_dim
        # self.proprio_encoder = Mlp(proprio_input_dim, proprio_hidden_dim, proprio_hidden_dim, norm_layer=nn.LayerNorm)

        # action decoder
        self.decoder_head = decoder_head
        if decoder_head == 'base':
            self.policy = BaseHead(num_blocks=policy_num_blocks, input_dim=self.vision_dim + self.proprio_dim, 
                                hidden_dim=policy_hidden_dim, action_size=action_size * chunk_length)
        elif decoder_head == 'ddpm':
            self.policy = DDPMHead(num_blocks=policy_num_blocks, input_dim=self.vision_dim + self.proprio_dim, 
                                hidden_dim=policy_hidden_dim, action_size=action_size * chunk_length)
        else :
            raise NotImplementedError

        # loss function
        self.loss_func = loss_func(**loss_func_conig)

    def forward_cond(self, instruction):
        lang_embeding = self.latent_proj.encode_text(instruction) # B 1024
        return lang_embeding

    def forward_obs(self, cur_images, cur_proprios, conditions):
        B, V, C, H, W = cur_images.shape
        cond = conditions.unsqueeze(1).repeat(1, V, 1).reshape(B*V, -1)
        vision_obs = cur_images.reshape(B*V, C, H, W) # B*2 3 224 224
        vision_semantics = self.vision_encoder(vision_obs, cond)
        vision_semantics = vision_semantics.reshape(B, -1)
        # proprio_semantics = self.proprio_encoder(cur_proprios)
        proprio_semantics = cur_proprios
        all_obs = torch.cat([vision_semantics, proprio_semantics], dim=-1)
        return all_obs
    
    def forward_policy(self, all_obs, cur_actions):
        if self.decoder_head == 'base':
            pred_actions = self.policy(all_obs)
            loss = self.loss_func(pred_actions, cur_actions)
            return loss
        elif self.decoder_head == 'ddpm':
            noise, t_tensor, noise_action, noise_pred = self.policy(all_obs, cur_actions)
            loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
            return loss

    def generate_policy(self, all_obs):
        if self.decoder_head == 'base':
            return self.policy.generate(all_obs)
        elif self.decoder_head == 'ddpm':
            return self.policy.generate(all_obs)

    def forward(self, cur_images, cur_proprios, cur_actions, instruction, **kwargs):
        conditions = self.forward_cond(instruction)
        all_obs = self.forward_obs(cur_images, cur_proprios, conditions)
        loss = self.forward_policy(all_obs, cur_actions)
        # Note: 'loss' must be in the dict
        return loss, dict(loss=loss)
    
    def generate(self, cur_images, cur_proprios, instruction, **kwargs):
        conditions = self.forward_cond(instruction)
        all_obs = self.forward_obs(cur_images, cur_proprios, conditions)
        pred_actions = self.generate_policy(all_obs)
        # Note: 'actions' must be in the dict
        return pred_actions, dict(actions=pred_actions)

    def state_dict(self, *args, **kwargs):
        model_dict = super().state_dict(*args, **kwargs)
        filtered_state_dict = {k: v for k, v in model_dict.items() if 'latent_proj' not in k}
        return filtered_state_dict


def bc_policy_res34_libero(chunk_length=6, **kwargs):
    latent_proj = DecisionNCE(model_path='assets/weights/DecisionNCE/DecisionNCE-RoboMutual')
    return BCPolicy(latent_proj, proprio_input_dim=9, proprio_hidden_dim=32, vision_backbone_name="resnet34",
                    policy_num_blocks=3, policy_hidden_dim=256, action_size=7, chunk_length=chunk_length)

def bc_policy_ddpm_res34_libero(chunk_length=6, **kwargs):
    latent_proj = DecisionNCE(model_path='assets/weights/DecisionNCE/DecisionNCE-RoboMutual')
    return BCPolicy(latent_proj, proprio_input_dim=9, proprio_hidden_dim=32, vision_backbone_name="resnet34", decoder_head='ddpm',
                    policy_num_blocks=3, policy_hidden_dim=256, action_size=7, chunk_length=chunk_length)

def bc_policy_ddpm_res34_calvin(chunk_length=6, **kwargs):
    latent_proj = DecisionNCE(model_path='assets/weights/DecisionNCE/DecisionNCE-RoboMutual')
    return BCPolicy(latent_proj, proprio_input_dim=7, proprio_hidden_dim=32, vision_backbone_name="resnet34", decoder_head='ddpm',
                    policy_num_blocks=3, policy_hidden_dim=256, action_size=7, chunk_length=chunk_length)