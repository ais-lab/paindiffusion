import torch
import torch.nn as nn

from torch import Tensor

from .components.adapter import ConvAdapter
from .components.adapter import LinearAdapter
from .components.adapter import ConvLinearAdapter
from .components.convolution import Upsample
from .components.convolution import Downsample
from .components.convolution import ConvTimeRes
from .components.convolution import conv_nd

from .components.embedding import TimeEmbedding
from .components.attention import EfficientAttention

from .utils.misc import exists
from .utils.misc import default

from einops import rearrange

from typing import List, Optional

class UNet(nn.Module):
    '''
        U-Net model as introduced in:
        "U-Net: Convolutional Networks for Biomedical Image Segmentation".
        It is a common choice as network backbone for diffusion models.         
    '''

    def __init__(
        self,
        net_dim : int = 4,
        in_dim : Optional[int] = None,
        out_dim : Optional[int] = None,
        attn_dim : int = 128,
        in_channels : int = 3,
        channels : int = 3,
        ctrl_dim : Optional[int] = None,
        ctrl_channels : int = 3,
        conv_type : int = 2, # 1D, 2D, 3D
        chn_mult : List[int] = (1, 2, 4, 8),
        num_group : int = 8,
        num_heads : int = 4,
        qry_chunk : int = 512,
        key_chunk : int = 1024,
    ) -> None:
        super().__init__()

        out_dim = default(out_dim, channels)

        self.channels = channels

        self.proj_inp_fix_channels = conv_nd(conv_type, in_channels, channels, 7, padding = 3) if in_channels != channels else nn.Identity()
        self.proj_inp_fix_dim = nn.Linear(in_dim, net_dim) if exists(in_dim) else nn.Identity()
        
        # NOTE: We need channels * 2 to accomodate for the self-conditioning
        self.proj_inp = conv_nd(conv_type, self.channels * 2, net_dim, 7, padding = 3)
        self.proj_ctrl = conv_nd(conv_type, ctrl_channels, net_dim, 1) if exists(ctrl_dim) else nn.Identity()
        # self.proj_inp = nn.Conv2d(self.channels, net_dim, 7, padding = 3)

        dims = [net_dim, *map(lambda m: net_dim * m, chn_mult)]
        mid_dim = dims[-1]

        dims = list(zip(dims, dims[1:]))

        # * Context embedding
        ctx_dim = net_dim * 4
        self.time_emb = nn.Sequential(
            TimeEmbedding(net_dim),
            nn.Linear(net_dim, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, ctx_dim)
        )

        # * Building the model. It has three main components:
        # * 1) The downsampling module
        # * 2) The bottleneck module
        # * 3) The upsampling module
        self.downs = nn.ModuleList([])
        self.ups   = nn.ModuleList([])
        num_resolutions = len(dims)

        attn_kw = {
            'num_heads' : num_heads,
            'qry_chunk' : qry_chunk,
            'key_chunk' : key_chunk,
            'pre_norm'  : True,
            # "key_dim": 512 => already set with controll dim
        }

        # qkv_adapt = ConvLinearAdapter if exists(ctrl_dim) else ConvAdapter
        qkv_adapt = LinearAdapter

        # Build up the downsampling module
        for idx, (dim_in, dim_out) in enumerate(dims):
            is_last = idx >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvTimeRes(dim_in, dim_in, ctx_dim = ctx_dim, num_group = num_group, conv_type=conv_type),
                ConvTimeRes(dim_in, dim_in, ctx_dim = ctx_dim, num_group = num_group, conv_type=conv_type),
                EfficientAttention(dim_in, attn_dim, qkv_adapt = qkv_adapt, key_dim = ctrl_dim, **attn_kw),
                conv_nd(conv_type, dim_in, dim_out, 3, padding = 1) if is_last else Downsample(dim_in, dim_out, conv_type=conv_type)
            ]))

        # Buildup the bottleneck module
        self.mid_block1 = ConvTimeRes(mid_dim, mid_dim, ctx_dim = ctx_dim, num_group = num_group, conv_type=conv_type)
        self.mid_attn   = EfficientAttention(mid_dim, attn_dim, qkv_adapt = qkv_adapt, key_dim = ctrl_dim, **attn_kw)
        self.mid_block2 = ConvTimeRes(mid_dim, mid_dim, ctx_dim = ctx_dim, num_group = num_group, conv_type=conv_type)

        # Build the upsampling module
        # NOTE: We need to make rooms for incoming residual connections from the downsampling layers
        for idx, (dim_in, dim_out) in enumerate(reversed(dims)):
            is_last = idx >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvTimeRes(dim_in + dim_out, dim_out, ctx_dim = ctx_dim, num_group = num_group, conv_type=conv_type),
                ConvTimeRes(dim_in + dim_out, dim_out, ctx_dim = ctx_dim, num_group = num_group, conv_type=conv_type),
                EfficientAttention(dim_out, attn_dim, qkv_adapt = qkv_adapt, key_dim = ctrl_dim, **attn_kw),
                conv_nd(conv_type,dim_out, dim_in, 3, padding = 1) if is_last else Upsample(dim_out, dim_in, conv_type=conv_type)
            ]))

        self.final = ConvTimeRes(net_dim * 2, net_dim, ctx_dim = ctx_dim, num_group = num_group, conv_type=conv_type)
        self.proj_out = conv_nd(conv_type, net_dim, out_dim, 1)

    def forward(
        self,
        imgs : Tensor,
        time : Tensor,
        x_c : Optional[Tensor] = None,
        ctrl : Optional[Tensor] = None,
    ) -> Tensor:
        '''
            Compute forward pass of the U-Net module. Expect input
            to be image-like and expects an auxilliary time signal
            (1D-like) to be provided as well. An optional contextual
            signal can be provided and will be used by the attention
            gates that will function as cross-attention as opposed
            to self-attentions.

            Params:
                - imgs: Tensor of shape [batch, channel, H, W] or [batch, d]
                
                - time: Tensor of shape [batch, 1]

                - context[optional]: Tensor of shape [batch, seq_len, emb_dim]

            Returns:
                - imgs: Processed images, tensor of shape [batch, channel, H, W]
        '''
        
        if len(imgs.shape) == 2: # If the input is 1D-like
            imgs = rearrange(imgs, 'b d -> b 1 d')
            
        # imgs = self.proj_inp_fix_channels(imgs)
        # imgs = self.proj_inp_fix_dim(imgs)
        # ctrl = self.proj_ctrl(ctrl)

        # Optional self-conditioning to the model
        cond = default(x_c, torch.zeros_like(imgs))
        imgs = torch.cat((cond, imgs), dim = 1)

        x = self.proj_inp(imgs)
        t = self.time_emb(time)

        h = [x.clone()]

        for conv1, conv2, cross_attn, down in self.downs:
            x = conv1(x, t)
            h += [x]

            x = conv2(x, t)
            x = cross_attn(x, ctrl)
            h += [x]

            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, ctrl)
        x = self.mid_block2(x, t)

        for conv1, conv2, cross_attn, up in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = conv1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = conv2(x, t)
            x = cross_attn(x, ctrl)

            x = up(x)

        x = torch.cat((x, h.pop()), dim = 1)

        x = self.final(x, t)

        return self.proj_out(x)