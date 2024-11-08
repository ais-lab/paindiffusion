import torch
from einops import repeat

mean_face = torch.load('baselines/mean_face.pt')

def mean_static_face( t):
    return repeat(mean_face.squeeze(), 'c -> t c', t=t)


