import torch.nn as nn
import torch.nn.functional as F

class MlpResNetBlock(nn.Module):
    """
    the MLPResnet Blocks used in IDQL: arXiv:2304.10573, Appendix G
    """
    def __init__(self, hidden_dim:int, ac_fn=F.gelu, use_layernorm=False, dropout_rate=0.1):
        super(MlpResNetBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ac_fn = ac_fn
        self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim)
            
    def forward(self, x):
        out = self.dropout(x)
        out = self.norm1(out)
        out = self.dense1(out)
        out = self.ac_fn(out)
        out = self.dense2(out)
        out = x + out
        return out

class MlpResNet(nn.Module):
    """
    the LN_Resnet used in IDQL: arXiv:2304.10573
    """
    def __init__(self, num_blocks:int, input_dim:int, hidden_dim:int, output_size:int, 
                 ac_fn=F.gelu, use_layernorm=True, dropout_rate=0.1):
        super(MlpResNet, self).__init__()
        
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_size)
        self.ac_fn = ac_fn
        self.mlp_res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_res_blocks.append(MlpResNetBlock(hidden_dim, ac_fn, use_layernorm, dropout_rate))
            
    def forward(self, x):
        out = self.dense1(x)
        for mlp_res_block in self.mlp_res_blocks:
            out = mlp_res_block(out)
        out = self.ac_fn(out)
        out = self.dense2(out)
        return out