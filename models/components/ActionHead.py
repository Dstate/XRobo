
from timm.layers import Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn

from .MlpResNet import MlpResNet

class LearnedPosEmb(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * 0.2)
            
    def forward(self, x):
        f = 2 * torch.pi * x @ self.kernel.T
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f
    
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, x_shape):
    '''
    align the dimention of alphas_cumprod_t to x_shape
    
    a: alphas_cumprod_t, B
    x_shape: B x F x F x F
    output: alphas_cumprod_t B x 1 x 1 x 1]
    '''
    b, *_ = a.shape
    return a.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDPMHead(nn.Module):
    def __init__(
        self, 
        action_size: int = 7,
        num_blocks:int = 3, 
        input_dim:int = 544, 
        hidden_dim: int = 256,
        time_dim: int = 32,
        num_timesteps: int = 25,
        time_hidden_dim: int=256
    ):
        super().__init__()
        self.time_dim = time_dim
        self.num_timesteps = num_timesteps
        self.time_hidden_dim = time_hidden_dim
        self.action_size = action_size
        self.time_process = LearnedPosEmb(1, time_dim)
        self.time_encoder = Mlp(time_dim, time_hidden_dim, time_hidden_dim, norm_layer=nn.LayerNorm)
        self.model = MlpResNet(num_blocks=num_blocks, 
                               input_dim=input_dim + action_size + time_hidden_dim,
                               hidden_dim=hidden_dim, output_size=action_size)
        self.init_ddpm()

    def init_ddpm(self, device='cuda'):
        self.betas = cosine_beta_schedule(self.num_timesteps).to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
    
    def forward_features(self, all_obs, t_tensor, noise_action):
        time_embedding = self.time_process(t_tensor.view(-1, 1))
        time_embedding = self.time_encoder(time_embedding)
        input_feature = torch.cat([all_obs, time_embedding, noise_action], dim=-1)
        noise_pred = self.model(input_feature)
        return noise_pred

    def forward(self, all_obs, cur_action):
        noise, t_tensor, noise_action = self.add_noise(cur_action)
        noise_pred = self.forward_features(all_obs, t_tensor, noise_action)
        return noise, t_tensor, noise_action, noise_pred
    
    def generate(self, all_obs):
        shape = (all_obs.shape[0], self.action_size)
        action = torch.randn(shape, device=all_obs.device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0], 1), t, device=all_obs.device)
            noise_pred = self.forward_features(all_obs, t_tensor, action)
            action = self.denoise(action, t_tensor, noise_pred)
        return action

    def add_noise(self, x0: torch.Tensor):
        B = x0.shape[0]
        noise = torch.randn_like(x0, device=x0.device)
        t = torch.randint(0, self.num_timesteps, (B, ), device=x0.device)
        xt = self.q_sample(x0, t, noise)
        return noise, t, xt
    
    def denoise(self, x, t_tensor, noise):
        x = self.p_sample(x, t_tensor, noise)
        return x

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        sample noisy xt from x0, q(xt|x0), forward process
        
        Input:
        x0: ground truth value, here x0 is typically the ground truth action
        t: timestep
        noise: noise
        
        Return: noisy samples
        """
        alphas_cumprod_t = self.alphas_cumprod[t]
        xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
        return xt
       
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor, guidance_strength=0, clip_sample=True, ddpm_temperature=1.):
        """
        sample xt-1 from xt, p(xt-1|xt)
        
        Input:
        xt: noisy samples, here xt is typically the noisy ground truth action
        t: timestep
        noise_pred: predicted noise
        guidance_strength: the strength of the guidance
        clip_sample: whether to clip the sample to [-1, 1]
        ddpm_temperature: the temperature of the noise
        
        Return: sample xt-1
        """
        alpha1 = 1 / torch.sqrt(self.alphas[t])
        alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
        
        xtm1 = alpha1 * (xt - alpha2 * noise_pred)
        
        noise = torch.randn_like(xtm1, device=xt.device) * ddpm_temperature
        xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
        
        if clip_sample:
            xtm1 = torch.clip(xtm1, -1., 1.)
        return xtm1

class BaseHead(nn.Module):
    def __init__(
        self, 
        action_size: int = 7,
        num_blocks:int = 3, 
        input_dim:int = 544, 
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_size = action_size
        self.model = MlpResNet(num_blocks=num_blocks, input_dim=input_dim,
                               hidden_dim=hidden_dim, output_size=action_size)
    
    def forward(self, all_obs):
        action = self.model(all_obs)
        return action
    
    def generate(self, all_obs):
        action = self.model(all_obs)
        return action