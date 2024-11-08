import math
from torch import nn
from torch.nn import functional as F
import torch
from einops import rearrange
from torch import einsum

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, is_temporal=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # if not is_temporal:
        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Conv1d(dim, hidden_dim, 1, bias = False)
        self.to_v = nn.Conv1d(dim, hidden_dim, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        # else:
        #     self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        #     self.to_k = nn.Linear(dim, hidden_dim, bias = False)
        #     self.to_v = nn.Linear(dim, hidden_dim, bias = False)
        #     self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, context = None):
        b, c, n = x.shape
        
        if exists(context):
            qkv = self.to_q(x), self.to_k(context), self.to_v(context)
        else:
            qkv = self.to_q(x), self.to_k(x), self.to_v(x)
            
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, is_temporal=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        
        # TODO: add a temporal or spatial option for the attention -> one work with the time and the other with the space
        # if not is_temporal:
        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Conv1d(dim, hidden_dim, 1, bias = False)
        self.to_v = nn.Conv1d(dim, hidden_dim, 1, bias = False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )
        # else:
        #     self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        #     self.to_k = nn.Linear(dim, hidden_dim, bias = False)
        #     self.to_v = nn.Conv1d(dim, hidden_dim, bias = False)
        #     self.to_out = nn.Sequential(
        #     nn.Linear(hidden_dim, dim),
        #     RMSNorm(dim)
        # )


    def forward(self, x, context = None):
        b, c, n = x.shape
        
        if exists(context):
            qkv = self.to_q(x), self.to_k(context), self.to_v(context)
        else:
            qkv = self.to_q(x), self.to_k(x), self.to_v(x)
        
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        
        if len(x.shape) == 0: # a number
            x = x.unsqueeze(0)
        
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0., is_linear_attn = False, context_channels = None, is_temporal=False):
        super().__init__()
        
        if is_linear_attn:
            self.attn = LinearAttention(dim, heads = heads, dim_head = dim_head, is_temporal=is_temporal)
        else:
            self.attn = Attention(dim, heads = heads, dim_head = dim_head, is_temporal=is_temporal)
            
        # prepare the channels for the context should have the same channels as the input
        self.context_proj = nn.Conv1d(context_channels, dim, 1) if exists(context_channels) else None
            
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context = None, scale_shift = None):
        
        context = self.context_proj(context) if exists(context) else None
        
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.attn(self.norm(x), context) + x
        x = self.dropout(x)
        return x
    
class ResAttentionBlock(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0., is_linear_attn = False, context_channels = None, is_temporal=False, time_emb_dim=None):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2) # to create scale and shift
        ) if exists(time_emb_dim) else None
        
        self.block1 = CrossAttentionBlock(dim, heads = heads, dim_head = dim_head, dropout = dropout, is_linear_attn = is_linear_attn, context_channels = context_channels, is_temporal=is_temporal)
        self.block2 = CrossAttentionBlock(dim, heads = heads, dim_head = dim_head, dropout = dropout, is_linear_attn = is_linear_attn, context_channels = context_channels, is_temporal=is_temporal)
        
    def forward(self, x, time_emb = None, context = None):
        
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> c b')
            scale_shift = time_emb.chunk(2, dim = 0)
            
        h = self.block1(x, context, scale_shift = scale_shift)
        h = self.block2(h, context)
        
        return x + h
    

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None, conds = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)
    
    
class ResnetBlock(nn.Module):
    
    # TODO: should we add the conditional input here?
    
    def __init__(self, dim, dim_out, *, time_emb_dim = None, conds_dim=None, dropout = 0.):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, conds = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
    
# class ConditionProjection(nn.Module):
#     def __init__(self, dim, channel, context_dim, context_channels):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(context_dim, dim) if dim!=context_dim else nn.Identity(),
#             nn.Conv1d(context_channels, channel, 1) if context_channels!=channel else nn.Identity()
#         )

#     def forward(self, x):
#         return self.proj(x)
    
# class MLP(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)
    
# class LayerNorm(nn.Module):
#     def __init__(self, dim, bias):
#         super().__init__()
        
#         self.weight = nn.Parameter(torch.ones(dim))
#         self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        
#     def forward(self, x):
#         return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    
# class AttnBlock(nn.Module):
#     def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0., bias = False):
#         super().__init__()

#         self.ln_1 = LayerNorm(dim, bias = bias)
#         self.attn = Attention(dim, heads = heads, dim_head = dim_head)
#         self.ln_2 = LayerNorm(dim, bias = bias)
#         self.mlp = MLP(dim, dim * 4, dropout)

#     def forward(self, x):
#         x = x + self.attn(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x
    
# class CrossAttnBlock(nn.Module):
#     def __init__(self, dim, heads, dim_head, dropout, bias=False):
#         super().__init__()

#         self.ln_1 = LayerNorm(dim, bias=bias)
#         self.self_attn = Attention(dim, heads=heads, dim_head=dim_head)
#         self.cross_attn = CrossAttention(dim, heads=heads, dim_heads=dim_head)
#         self.ln_2 = LayerNorm(dim, bias=bias)
#         self.ln_3 = LayerNorm(dim, bias=bias)
#         self.mlp = MLP(dim, dim * 4, dropout)
        
#         self.gated_alpha = nn.Parameter(torch.tensor(0.0))
        
#     def forward(self, x, y):
#         x = x + self.self_attn(self.ln_1(x))
#         x = x + self.gated_alpha*self.cross_attn(self.ln_2(x), self.ln_3(y))
#         x = x + self.mlp(self.ln_2(x))
#         return x, y