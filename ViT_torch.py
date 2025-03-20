import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import math

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torchvision
import matplotlib.pyplot as plt


print(torch.__version__, torchvision.__version__)
print(torch.cuda.is_available())  
print(torch.version.cuda)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Vit in Pytorch
from einops import rearrange
from einops.layers.torch import Rearrange

#import torchvision.transforms as transforms


def extract_patches_1D(pattern, patch_size):
    patches = pattern.unfold(-1, patch_size, patch_size)

    return patches

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        
        self.data = torch.tensor(X, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.float32)
        self.transform = transform

    def __len__(self): # I can use len method len(Dataset) --> overload of len(*)
        return len(self.data)

    def __getitem__(self, index): # I can use Dataset[2] --> overload of [*]
        x = self.data[index]
        y = self.label[index]
        
        if self.transform:
            x = self.transform(x)

        return x, y

class Patching_PositionalEncoding(nn.Module):
    def __init__(self, patch_size=10, max_len=4500):
        super(Patching_PositionalEncoding, self).__init__()
        self.patch_size = patch_size
        
        # Create a long enough 'position' tensor
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, patch_size, 2).float() * -(math.log(10000.0) / patch_size))
        

        pe = torch.zeros(max_len, patch_size)
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        # inputs in BATCH_SIZE, TOKENS, DIM

        x = torch.split(x, self.patch_size, dim=1)  # Returns a tuple of 450 tensors of shape [32, 10]
        x = torch.stack(x, dim=1).squeeze(-1)
        seq_len = x.size(1)

        return x + self.pe[:seq_len, :].unsqueeze(0)  # Broadcast across batch dimension
    
class Project_Tokens(nn.Module):
    def __init__(self, patch_dim = 10, d_model = 64, num_tokens = 450):
        super(Project_Tokens, self).__init__()
        
        self.proj_matrix = nn.Parameter(torch.randn(patch_dim, d_model))   # tokens = 450 x 10  *  matrix = 10 x 64 -->  450 x 64
    
    def forward(self, x):
        return x @ self.proj_matrix

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    
    def __init__(self, dim, heads = 4, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attention = nn.Softmax(dim = -1)

        self.qvk = nn.Linear(dim, inner_dim*3, bias= False) # factor 3 is used then to "create" the Q, V and  K terms 
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        
        x = self.norm(x)
        qvk_ = self.qvk(x).chunk(3, dim = -1) # Splitted on the last axis

        q, v, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qvk_)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attention(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FFN(dim, mlp_dim)
                ]))
    def forward(self, x):

        for i in self.layers:
            
            x = i[0](x) + x
            x = i[1](x) + x
        return self.norm(x)

class ViT_1D(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_out, dim, depth, heads, mlp_dim, channels = 3, dim_heads = 32, mlp_head = 32, thr_en = False, thr =100):
        super().__init__()

        num_patches = seq_len // patch_size
        patch_dim =  patch_size


        self.thr_cond = thr_en
        self.threshold = thr


        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.patching_pe = Patching_PositionalEncoding(patch_size = patch_size)

        self.proj_tokens = Project_Tokens(patch_dim = patch_size, d_model = dim, num_tokens=num_patches)

        self.transformer = Transformer(dim, depth, heads, dim_heads, mlp_dim)

        self.to_latent = nn.Identity()

        self.mpl_headNN = nn.Linear(dim, mlp_head)

        self.linear = nn.Linear(mlp_head, num_out)

    def forward(self, series):
        if self.thr_cond:
            series[series>self.threshold] = 1000


        #print("Input: ",series.shape)
        x = self.patching_pe(series)
        #print("patching_pe: ",x.shape)

        x = self.proj_tokens(x)

        #print("Output patching: ", x.shape)

        x = self.transformer(x)
        #print("transformer : ", x.shape)


        #x = x.permute(1, 0, 2)
        x = x.mean(dim = 1)
        #print("mean : ", x.shape)


        x = self.to_latent(x)
        x = self.mpl_headNN(x)
        #print("mpl_headNN", x.shape)
        x = self.linear(x)

        #print("linear", x.shape)
        return x
