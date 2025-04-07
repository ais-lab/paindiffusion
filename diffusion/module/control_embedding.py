from typing import Dict, Optional, Union
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ControlEmbedding(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        in_dim: int=1,
        conf: Optional[Dict[str, Union[str, int]]] = None,
    ) -> None:
        super().__init__()

        self.emotion = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )
        self.pain_expressive = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )
        self.stimulus_abs = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )
        self.conf = conf
        
        self.null_ctrl = lambda ctrl : torch.zeros_like(ctrl)
        
    def forward(self, ctrl, is_training=False):
        if len(ctrl) == 4:
            control_emotion, pain_expressiveness, stimulus_abs, _ = ctrl
        elif len(ctrl) == 3:
            control_emotion, pain_expressiveness, stimulus_abs = ctrl
        
        control_emotion = (control_emotion - self.conf['emotion']['mean']) / self.conf['emotion']['std']
        pain_expressiveness = (pain_expressiveness - self.conf['expressiveness']['mean']) / self.conf['expressiveness']['std']
        stimulus_abs = (stimulus_abs - self.conf['stimulus_abs']['mean']) / self.conf['stimulus_abs']['std']

        control_emotion, pain_expressiveness, stimulus_abs = map(
            lambda x: rearrange(x, "b ... -> b ... 1").contiguous(),
            [control_emotion, pain_expressiveness, stimulus_abs],
        )       
        
        if is_training and random() < 0.3:
            random_length = torch.randint(1, stimulus_abs.shape[1], (1,)).item()
            stimulus_abs = stimulus_abs[:, :random_length]
            
        control_emotion = self.emotion(control_emotion)
        pain_expressiveness = self.pain_expressive(pain_expressiveness)
        stimulus_abs = self.stimulus_abs(stimulus_abs) 
        
        fill_sequence_stimulus = control_emotion.shape[1] - stimulus_abs.shape[1]
        
        if fill_sequence_stimulus > 0:
            stimulus_abs = F.pad(stimulus_abs, (0,0,0,0,fill_sequence_stimulus,0), "constant", 0)
        
        out = torch.stack([control_emotion, pain_expressiveness, stimulus_abs])
        
        return out
    
    def dropout_ctrl(self, ctrl, drop_probs):

        num_controll = 3
        if isinstance(drop_probs, float):
            drop_probs = [drop_probs] * num_controll
        else:
            assert len(drop_probs) == num_controll, "The length of drop_prob should be equal to the number of control signal"

        if len(ctrl) == 4: # control_emotion, pain_expressiveness, stimulus_abs, pspi_no_au43 (pspi was added for evaluation)
            # drop the pspi_no_au43
            ctrl = ctrl[:-1]

        for idx, control_signal in enumerate(ctrl):
            if random() < drop_probs[idx]:
                ctrl[idx] = self.null_ctrl(control_signal)
                
        return ctrl