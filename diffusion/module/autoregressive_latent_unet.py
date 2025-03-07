from dataclasses import dataclass, field
from functools import partial
from typing import List

from einops import rearrange
from einops import reduce
import torch
from torch import nn

from .components.modules import (
    Downsample,
    RandomOrLearnedSinusoidalPosEmb,
    ResnetBlock,
    SinusoidalPosEmb,
    Upsample,
    default,
    CrossAttentionBlock,
    # ConditionProjection,
    exists,
)

from huggingface_hub import PyTorchModelHubMixin



# @dataclass
# class UNet1DConfig:
#     channels: int = 1  # in our case the channel is 1
#     # how deep the unet dim actually increase
#     dim_mults: List = field(default_factory=lambda: (1, 2, 4, 8))

#     # for flexibilty in terms of input/output dim
#     # working dim of unet
#     dim: int = 144
#     # input dim if different from working dim, we can cast from the dim to init dim
#     init_dim: int = None
#     # this is just the same as dim, if no change in the usage module
#     out_dim: int = None

#     # context dim for the attention
#     # conditions: temperature, average pain tolerance, average emotion (each is a number -> 1 channel)
#     # so the context_channels is 3
#     context_input_dim: int = 1
#     context_channels: int = 1
#     context_dim: int = 128
    
#     # past block size
#     past_block_size: int = 128

#     # for regularization
#     drop_out: float = 0.0

#     # increase performance of diffusion
#     self_condition: bool = True
#     learned_variance: bool = False

#     # time - diffusion - encoding
#     learned_sinusoidal_cond: bool = False
#     random_fourier_features: bool = False
#     learned_sinusoidal_dim: int = 16
#     sinusoidal_pos_emb_theta: int = 10000

#     # attention layer config
#     attn_dim_head: int = 32
#     attn_heads: int = 4


# TODO: change to video latent unet


class LatentUnet(nn.Module,
                     PyTorchModelHubMixin,
):

    def __init__(self, 
                 dim: int,
                    context_channels: int,
                    context_dim: int,
                    context_input_dim: int,
                    drop_out: float,
                    self_condition: bool,
                    learned_variance: bool,
                    learned_sinusoidal_cond: bool,
                    random_fourier_features: bool,
                    learned_sinusoidal_dim: int,
                    sinusoidal_pos_emb_theta: int,
                    attn_dim_head: int,
                    attn_heads: int,
                    dim_mults: List[int],
                    channels: int,
                    out_dim: int = None,
                    in_dim: int = None,
                    init_dim: int = None,
                 ):
        super().__init__()

        # determine the dimension
        
        self.channels = channels
        
        self.dim = dim
        
        self.context_channels = context_channels
        self.context_dim = context_dim
        self.self_condition = self_condition
       
        input_channels = channels * (2 if self.self_condition else 1)
        
        # how many downs level

        # each block should contain what

        # refer to the https://github.com/CompVis/latent-diffusion to implement the cross attention for the condition

        init_dim = default(init_dim, dim)
        
        self.context_proj = nn.Sequential(
            # nn.Conv1d(context_channels, context_channels, 1),
            # nn.Linear(context_input_dim, context_dim),
            # nn.GELU(),
            # nn.Linear(context_dim, context_dim),
            nn.Identity()
        )

        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4  # why

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        resnet_block = partial(
            ResnetBlock, time_emb_dim=time_dim, dropout=drop_out
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (len(in_out) - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        # Done: turn linear attention into linear cross attention
                        # investigate why they use linear attention here -> linear attention is used to lower computational cost
                        # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        CrossAttentionBlock(
                            dim_in,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            is_linear_attn=True,
                            context_channels=context_channels, # context + past as well
                        ),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        # Done: change the attention to takein the cross modality
        self.mid_attn = CrossAttentionBlock(
            mid_dim,
            dim_head=attn_dim_head,
            heads=attn_heads,
            is_linear_attn=False,
            context_channels=context_channels,
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        # why this block input is dim_in + dim_out: because it have the resnet connection from down blocks
                        resnet_block(dim_in + dim_out, dim_out),
                        resnet_block(dim_in + dim_out, dim_out),
                        # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        CrossAttentionBlock(
                            dim_out,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            is_linear_attn=True,
                            context_channels=context_channels,
                        ),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv1d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward(self, x: torch.Tensor, time, x_c=None, ctrl=None):
        # x : b, c, d
        # b, c, d = x.shape
        
        # if len(x.shape) == 2:
        #     x = rearrange(x, 'b d -> b 1 d')
        
        # if exists(x_c):
        #     if len(x_c.shape) == 2:
        #         x_c = rearrange(x_c, 'b d -> b 1 d')
            
        b, c, d = x.shape
        
        # using self condition to increase the generation quality
        x_c = default(x_c, lambda: torch.zeros_like(x))
        x = torch.cat((x_c, x), dim=1)
            
        x = self.init_conv(x)

        r = x.clone()

        t = self.time_mlp(
            time
        )  # to turn (time of diffusion process) into feature vector then add
            
        if exists(ctrl):
            ctrl = self.context_proj(ctrl)
        else:
            ctrl = torch.zeros((b, self.context_channels, self.context_dim))
            
        # skip connection placeholder
        h = []

        # down of unet
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x, ctrl)
            h.append(x)

            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, ctrl)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x)
            x = attn(x, ctrl)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        # if c == 1:
        #     x = reduce(x, "b c d -> b d", "max")
        
        return x


# @dataclass
# class ConditionedTransformerConfig:
#     input_dim: int=128
#     hidden_dim: int=256
#     output_dim: int=128


# class ConditionedTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         self.proj_in = nn.Linear(self.input_dim, self.hidden_dim)

#         self.attention_block = nn.Sequential(
#             [CrossAttnBlock(dim=self.hidden_dim, heads=attn_heads, dim_head=attn_dim_head) for _ in range(attn_layers)]
#         )

#         self.proj_out = nn.Linear(self.hidden_dim, self.output_dim)

#     def forward(self, x, time, x_self_cond=None, conds=None):
#         pass

if __name__ == "__main__":


    model = LatentUnet(
        dim=128,
        context_channels=3,
        context_dim=128,
        context_input_dim=1,
        past_block_size=128,
        drop_out=0.0,
        self_condition=True,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        dim_mults=[1, 2, 4, 8],
        channels=1,
        # out_dim=128
                   
                   )

    x = torch.randn(5, 1, 128)

    temp_cond = torch.randn(5, 1)
    avg_pain_tolerence_cond = torch.randn(5, 1)
    avg_emotion_cond = torch.randn(5, 1)

    # the past frame of the video
    past_x = torch.randn(5, 128, 128) # B, block_size, dim

    conds = torch.cat([temp_cond, avg_pain_tolerence_cond, avg_emotion_cond], dim=1)
    conds = conds.unsqueeze(2)
    # conditions: B, 3, 1
    
    time = torch.randint(0, 100, (5,))

    print(model(x, time).shape)
    print(model(x, time, x_self_cond=x).shape)
    print(model(x, time, x_self_cond=x, conds=conds).shape)
    print(model(x, time, conds=conds).shape)
