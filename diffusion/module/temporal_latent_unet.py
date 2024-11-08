# adapted from lucidrains/denoising-diffusion-pytorch-1d

import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from .components.modules import (
    Attention,
    Downsample,
    LinearAttention,
    PreNorm,
    RandomOrLearnedSinusoidalPosEmb,
    Residual,
    ResnetBlock,
    SinusoidalPosEmb,
    Upsample,
    default,
    CrossAttentionBlock,
    # ConditionProjection,
    exists,
    ResAttentionBlock
)

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

    
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, context=None, *args,  **kwargs):
        shape = x.shape
        
        kwargs_for_einops = kwargs.pop('kwargs_for_einops', {})
        
        # reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}', **kwargs_for_einops)
        context = rearrange(context, f'{self.from_einops} -> {self.to_einops}', **kwargs_for_einops) if exists(context) else None
        x = self.fn(x, context=context, *args, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **kwargs_for_einops)
        return x 


# model

# TODO: add skip connection for temporal attention with a weighted learned parameter

class Unet1D(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        self_condition = False,
        learned_variance = False,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        
        context_channels=4,
        context_dim=1,  
        time_window=64
    ):
        super().__init__()

        # determine dimensions
        
        # define temporal attn and temporal conv
        

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_noise_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
        fourier_dim = dim

        self.time_noise_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_noise_dim),
            nn.GELU(),
            nn.Linear(time_noise_dim, time_noise_dim)
        )
        
        time_temporal_dim = dim * 4
        
        time_temporal_pos_emb = SinusoidalPosEmb(time_temporal_dim, theta = sinusoidal_pos_emb_theta)
        self.time_temporal_pos_emb = nn.Sequential(
            time_temporal_pos_emb,
            nn.Linear(time_temporal_dim, time_temporal_dim),
            nn.GELU(),
            nn.Linear(time_temporal_dim, time_temporal_dim)
        )

        resnet_block = partial(ResnetBlock, time_emb_dim = time_noise_dim, dropout = dropout)
        temporal_attn = lambda dim, is_linear: EinopsToAndFrom( '(b t) c d', '(b d) c t', 
                                ResAttentionBlock(
                                    dim,
                                    dim_head=attn_dim_head,
                                    heads=attn_heads,
                                    context_channels=context_channels, # this is for condition proj
                                    # context_dim=context_dim,
                                    is_linear_attn=is_linear,
                                    dropout=dropout,
                                    is_temporal=True,
                                    time_emb_dim=time_temporal_dim,
                                ))
        
        
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Linear(dim, dim)
        )
                                
        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                CrossAttentionBlock(
                            dim_in,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            is_linear_attn=True,
                            context_channels=context_channels,
                        ),
                temporal_attn(dim_in, is_linear = False),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_spatial_attn = CrossAttentionBlock(
                            mid_dim,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            is_linear_attn=False,
                            context_channels=context_channels,
                        )
        self.mid_temporal_attn = temporal_attn(mid_dim, is_linear = False)
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                CrossAttentionBlock(
                            dim_out,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            is_linear_attn=True,
                            context_channels=context_channels,
                        ),
                temporal_attn(dim_out, is_linear = False),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward(self, x, time_noise, x_c = None, ctrl = None):
        
        b, t, c, d = x.shape

        # temporal time bias 
        self.time_temporal_pos = torch.arange(0, t, dtype=torch.float32, device=x.device)
                
        if exists(ctrl):
            # context = self.context_mlp(context)
            ctrl = rearrange(ctrl, 'b t c d -> (b t) c d', b=b)
        else:
            ctrl = None
        
        x = rearrange(x, 'b t c d -> (b t) c d')
        
        time_noise = rearrange(time_noise, 'b t -> (b t)', t = t)
        
        temporal_positional_bias = self.time_temporal_pos_emb(self.time_temporal_pos)
        
        if self.self_condition:
            x_c = rearrange(x_c, 'b t c d -> (b t) c d') if exists(x_c) else None
            x_c = default(x_c, lambda: torch.zeros_like(x))
            x = torch.cat((x_c, x), dim = 1)

        x = self.init_conv(x)
        # x = self.init_temporal_attn(x, kwargs_for_einops = {'b': b, 'temporal': t})
        
        r = x.clone()

        t_noise = self.time_noise_mlp(time_noise)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t_noise)
            h.append(x)

            x = block2(x, t_noise)
            x = spatial_attn(x, ctrl)
            x = temporal_attn(x, 
                            #   context=context, # because d change, we cannot add the context here
                              time_emb = temporal_positional_bias, kwargs_for_einops = {'b': b, 't': t})
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_noise)
        x = self.mid_spatial_attn(x, ctrl)
        x = self.mid_temporal_attn(x, 
                                #    context, 
                                   kwargs_for_einops = {'b': b, 't': t}, time_emb = temporal_positional_bias)
        x = self.mid_block2(x, t_noise)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t_noise)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t_noise)
            x = spatial_attn(x, ctrl)
            x = temporal_attn(x, 
                            #   context, 
                              kwargs_for_einops = {'b': b, 't': t}, time_emb = temporal_positional_bias)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t_noise)
        x = self.final_conv(x)
        
        x = rearrange(x, '(b t) c d -> b t c d', b=b)
        return x


if __name__ == "__main__":
    
    batch = 13
    time_window = 64
    channels = 3
    dim = 128
    
    context_channels = 4
    context_dim = 1
    
    x = torch.randn(batch, time_window, channels, dim)
    time = torch.randint(0, 35, (batch,)) # noise time step
    
    context = torch.randn(batch, time_window, context_channels, context_dim)
    
    model = Unet1D(dim, channels = channels, self_condition = True,
                   context_channels=context_channels, context_dim=context_dim, time_window=time_window)
    
    out = model(x, time)
    print(out.shape)
    
    out = model(x, time, x)
    print(out.shape)
    
    out = model(x, time, ctrl = context)
    print(out.shape)
    
    out = model(x, time, x, context)
    print(out.shape)