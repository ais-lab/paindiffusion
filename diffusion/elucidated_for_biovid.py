from random import random
import time
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import AdamW
from .diffusion import Diffusion

from .module.utils.misc import exists
from .module.utils.misc import default
from .module.utils.misc import enlarge_as

from typing import Callable, Dict, Optional, Union
from einops import rearrange, reduce

# TODO: sample multiple frames at once to speed up the sampling process

class RelativePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_position * 2 + 1, emb_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q):
        range_vec_q = torch.arange(seq_len_q, device=self.embeddings_table.device)
        clipped_relative_matrix = torch.clamp(range_vec_q, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings

class ControlEmbedding(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        in_dim: int=1,
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
        
        self.relative_time_emb = RelativePositionalEncoder(128, 256)
        

    def forward(self, ctrl, is_training=False):

        # [B, sequence_len], [B, sequence_len], [B, sequence_len], [B, sequence_len, latent_dim]
        control_emotion, pain_expressiveness, stimulus_abs, past_latent = ctrl

        control_emotion, pain_expressiveness, stimulus_abs = map(
            lambda x: rearrange(x, "b s -> b s 1").contiguous(),
            [control_emotion, pain_expressiveness, stimulus_abs],
        )       
        
        # fill 0 to the past_emb to match the sequence length
        
        if is_training and random() < 0.3: # crop at random length
            random_length = torch.randint(1, past_latent.shape[1], (1,)).item()
            past_latent = past_latent[:, :random_length, :]
            stimulus_abs = stimulus_abs[:, :random_length]
            
        control_emotion = self.emotion(control_emotion)
        pain_expressiveness = self.pain_expressive(pain_expressiveness)
        stimulus_abs = self.stimulus_abs(stimulus_abs) 
        
        fill_sequence_latent = control_emotion.shape[1] - past_latent.shape[1]
        fill_sequence_stimulus = control_emotion.shape[1] - stimulus_abs.shape[1]
        past_latent = F.pad(past_latent, (0,0,fill_sequence_latent,0), "constant", 0)
        stimulus_abs = F.pad(stimulus_abs, (0,0,fill_sequence_stimulus,0), "constant", 0)
        
        past_latent = past_latent + self.relative_time_emb(past_latent.shape[1])
        
        # target shape: [B, sequence_lenth, emb_dim * num_ctrl]
        # out = torch.cat([control_emotion, pain_expressiveness, stimulus_abs, past_latent], dim=-1)
        out = torch.stack([control_emotion, pain_expressiveness, stimulus_abs, past_latent]) # [num_ctrl, B, sequence_lenth, emb_dim]
        return out

class ElucidatedDiffusion(Diffusion):
    """
    Denoising Diffusion Probabilistic Model as introduced in:
    "Elucidating the Design Space of Diffusion-Based Generative
    Models", Kerras et al. (2022) (https://arxiv.org/pdf/2206.00364)
    """


    def __init__(
        self,
        model: Module = None,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        ode_solver: str = "heun_sde",
        rho_schedule: float = 7,
        lognorm_mean: float = -1.2,
        lognorm_std: float = +1.2,
        sigma_data: float = 0.5,
        sample_output_dir: str = None,
        drop_probs: Union[float, list] = 0.0,
        guide: Union[float, list] = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            ode_solver=ode_solver,
            **kwargs,
        )
        
        # Randomly drop the control signal
        self.drop_probs = drop_probs
        
        self.guide = guide

        # Controls for timesteps and schedule generation
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho_schedule = rho_schedule

        # Controls for training
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std
        self.sigma_data = sigma_data

        self.log_img_key = f"Eludidated Diffusion - {ode_solver}"

        self.norm_forward = lambda x: 2.0 * x - 1.0
        self.norm_backward = lambda x: 0.5 * (1 + x.clamp(-1.0, 1.0))

        self.ctrl_emb: Optional[nn.Module] = ControlEmbedding(
            emb_dim=128, in_dim=1
        )
        
        self.sample_output_dir = sample_output_dir
        
        self.save_hyperparameters(ignore=['model'])

    @torch.no_grad()
    def follow(
        self,
        *args,
        ctrl: Optional[Tensor] = None,
        guide: list = [
            1.0,
        ],
        **kwargs,
    ):
        """
        Implements Classifier-Free guidance as introduced
        in Ho & Salimans (2022).
        """
        if not exists(ctrl):
            return self.predict(*args, ctrl=ctrl, **kwargs)
        
        # DONE: change the guildance from different kind of signals, with each signal being control by difference guildance
        # maybe with a random function to turn on and off the guidance for each signal

        # Implementing the guidance for each signal
        # follow http://arxiv.org/abs/2404.10667
        # NOTE: lenght of the guilde should be equal to the number of signals
        
        num_controll = 4
        batch_size = ctrl.shape[1]
        
        assert len(guide) == num_controll
        
        original_ctrl = ctrl.clone()

        # cond = self.predict(*args, ctrl=ctrl, **kwargs)
        sum_scale = sum(guide)
                
        # bring the controll signal to the front to iterate over
        # _ctrl = rearrange(original_ctrl, "b l (d c) -> c b l d", c=num_controll)

        # zero out the control signal
        null_ctrl = [rearrange(original_ctrl.clone(), "c b l d -> b l (c d)").contiguous()]
        for idx, control_signal in enumerate(original_ctrl):
            _null_ctrl = self.null_ctrl(control_signal)

            with_null_ctrl = original_ctrl.clone()
            with_null_ctrl[idx] = _null_ctrl
            
            # bring the controll signal back to the back to pass to the model
            with_null_ctrl = rearrange(with_null_ctrl, "c b l d -> b l (c d)").contiguous()
            
            null_ctrl.append(with_null_ctrl)
            
        ctrl = torch.cat(null_ctrl, dim=0) # ((num_controll+1)*batch, len, dim)

        # repeat the input to match the batch size
        new_args = [input.clone().repeat_interleave(num_controll+1, dim=0) if len(input.shape)>1 else input for input in args]
        if exists(kwargs['x_c']):
            x_c = kwargs['x_c']
            x_c = x_c.clone().repeat_interleave(num_controll+1, dim=0)
            kwargs['x_c'] = x_c
        
        # predict in batch for effiency
        preds = (
            self.predict(*new_args, ctrl=ctrl, **kwargs)
        )
        
        cond = preds[:batch_size]
        
        null = torch.zeros_like(cond)
        for idx in range(1, num_controll+1):
            sub_null = preds[idx*batch_size:(idx+1)*batch_size] * guide[idx-1]
            null += sub_null
            
        output = (1 + sum_scale) * cond - null
        return output


    def predict(
        self,
        x_t: Tensor,
        sig: Tensor,
        x_c: Optional[Tensor] = None,
        ctrl: Optional[Tensor] = None,
        clamp: bool = False,
    ) -> Tensor:
        """
        Apply the backbone model to come up with a prediction, the
        nature of which depends on the diffusion objective (can either
        be noise|x_start|v prediction).
        """

        bs, *_, device = x_t.shape, x_t.device

        if isinstance(sig, float):
            sig = torch.full((bs,), sig, device=device)

        # merge the control signal here

        # Inject appropriate noise value to images
        p_sig = enlarge_as(sig, x_t)
        x_sig = self.c_in(p_sig) * x_t
        t_sig = self.c_noise(sig)

        # Use the model to come up with a (hybrid) prediction the nature of
        # which depends on the implementation of the various c_<...> terms
        # so that the network can either predict the noise (eps) or the
        # input directly (better when noise is large!)
        out: Tensor = self.model(x_sig, t_sig, x_c=x_c, ctrl=ctrl)
        out: Tensor = self.c_skip(p_sig) * x_t + self.c_out(p_sig) * out

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    @torch.no_grad()
    def forward(
        self,
        num_imgs: int = 4,
        num_steps: Optional[int] = None,
        ode_solver: Optional[str] = None,
        norm_undo: Optional[Callable] = None,
        ctrl: Optional[Tensor] = None,
        use_x_c: Optional[bool] = None,
        guide: Union[float, list]= 1.,
        is_training: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Sample images using a given sampler (ODE Solver)
        from the trained model.
        """

        use_x_c = default(use_x_c, self.self_cond)
        num_steps = default(num_steps, self.sample_steps)
        norm_undo = default(norm_undo, self.norm_backward)
        self.ode_solver = default(ode_solver, self.ode_solver)

        timestep = self.get_timesteps(num_steps)
        schedule = self.get_schedule(timestep)
        scaling = self.get_scaling(timestep)

        # schedule = repeat(schedule, '... -> b ...', b = num_imgs)
        # scaling  = repeat(scaling , '... -> b ...', b = num_imgs)

        # Encode the condition using the sequence encoder
        ctrl = self.ctrl_emb(ctrl, is_training=is_training)[:,:num_imgs] if exists(ctrl) else ctrl
        # ctrl = rearrange(ctrl, "c b l d -> b l (c d)").contiguous() if exists(ctrl) else ctrl

        # FIXME: fix the shape for sampling => because it's latent
        shape = (num_imgs, 1, 128)

        x_0 = self.sampler(
            shape, schedule, scaling, ctrl=ctrl, use_x_c=use_x_c, guide=guide, **kwargs
        )

        return norm_undo(x_0)
    
    def compute_loss(
        self,
        x_0 : Tensor,
        ctrl : Optional[Tensor] = None,
        use_x_c : Optional[bool] = None,     
        norm_fn : Optional[Callable] = None,
        is_training : bool = False,
    ) -> Tensor:

        use_x_c = default(use_x_c, self.self_cond)
        norm_fn = default(norm_fn, self.norm_forward)
        
        # get the previous frame in temporal from control signal
        # if exists(ctrl):
        #     prev_frame = ctrl[3][:,-10:]
            
        #     prev_frame = reduce(prev_frame, "b l d -> b d", "mean")
             
        #     prev_frame = rearrange(prev_frame, "b d -> b 1 d") 
            
        #     # check if prev contain all zeros
        #     if torch.all(prev_frame == 0):
        #         prev_frame = None
            
        # else:
        #     prev_frame = None
        
        # coherence_weight = 0.5

        # Encode the condition using the sequence encoder
        ctrl = self.ctrl_emb(ctrl, is_training) if exists(ctrl) else ctrl
        ctrl = rearrange(ctrl, "c b l d -> b l (c d)").contiguous() if exists(ctrl) else ctrl

        # Normalize input images
        x_0 = norm_fn(x_0)
        # prev_frame = norm_fn(prev_frame) if exists(prev_frame) else prev_frame

        bs, *_ = x_0.shape

        # Get the noise and scaling schedules
        sig = self.get_noise(bs)

        # NOTE: What to do with the scaling if present?
        # scales = self.get_scaling()

        eps = torch.randn_like(x_0)
        x_t = x_0 + enlarge_as(sig, x_0) * eps # NOTE: Need to consider scaling here!

        x_c = None

        # Use self-conditioning with 50% dropout
        if use_x_c and random() < 0.5:
            with torch.no_grad():
                x_c = self.predict(x_t, sig, ctrl = ctrl)
                x_c.detach_()

        x_p = self.predict(x_t, sig, x_c = x_c, ctrl = ctrl)

        # Compute the reconstruction loss
        loss = self.criterion(x_p, x_0, reduction = 'none')
        
        # add a comparison with the previous frame to increase the temporal coherence
        # coherence_loss = coherence_weight*(self.criterion(x_p, prev_frame, reduction = 'none') if exists(prev_frame) else 0)
        # loss = loss + coherence_loss
                
        loss : Tensor = reduce(loss, 'b ... -> b', 'mean')

        # Add loss weight
        loss *= self.loss_weight(sig)
        return loss.mean()
    
     # * Lightning Module functions
    def training_step(self, batch : Dict[str, Tensor], batch_idx : int) -> Tensor:
        # Extract the starting images from data batch
        x_0  = batch[self.data_key]
        ctrl = batch[self.ctrl_key] if exists(self.ctrl_key) else None
        
        # DONE: random drop the control signal
        num_controll = 4
        if isinstance(self.drop_probs, float):
            self.drop_probs = [self.drop_probs] * num_controll
        else:
            assert len(self.drop_probs) == num_controll, "The length of drop_prob should be equal to the number of control signal"
        
        for idx, control_signal in enumerate(ctrl):
            if random() < self.drop_probs[idx]:
                ctrl[idx] = self.null_ctrl(control_signal)

        loss = self.compute_loss(x_0, ctrl = ctrl, is_training = True)

        self.log_dict({'train_loss' : loss}, logger = True, on_step = True, sync_dist = True)
        
        # if self.global_step < self.warmup_iters:
        #     _, scheduler = self.lr_schedulers()
        #     scheduler.step()

        return loss
    
    def validation_step(self, batch : Dict[str, Tensor], batch_idx : int) -> Tensor:
        # Extract the starting images from data batch
        x_0  = batch[self.data_key]
        ctrl = batch[self.ctrl_key] if exists(self.ctrl_key) else None
        
        # ctrl = self.ctrl_emb(ctrl) if exists(ctrl) else ctrl

        loss = self.compute_loss(x_0, ctrl = ctrl, is_training = False)

        self.log_dict({'val_loss' : loss}, logger = True, on_step = True, sync_dist = True)

        self.val_outs = (x_0, ctrl)
        
        return x_0, ctrl
    
    @torch.no_grad()
    def on_validation_epoch_end(self, 
                                # val_outs : Tuple[Tensor, ...]
                                ) -> None:
        '''
            At the end of the validation cycle, we inspect how the denoising
            procedure is doing by sampling novel images from the learn distribution.
        '''
        
        # DONE: sample a video instead of an image and call the renderer
        
        # TODO: investigate the bottleneck of the validation sample
        # maybe it comes with the inference engine of the lightning module?
        # run a profiller to figure out the bottleneck
        # if not able to fix it then just sample a bunch of frames instead of sampling one by one

        # self.sample_imgs()
        
        if self.trainer.global_step == 0:
            return
        
        self.sample_video()
        # pass
        
    @torch.no_grad()
    def sample_video(self, num_video=1, block_size=256):
        (x_0, ctrl) = self.val_outs
        
        # TODO: sample a longer video
        
        for batch_idx in range(num_video):
            control_emotion = ctrl[0][batch_idx] # (block_size,)
            pain_expressiveness = ctrl[1][batch_idx] # (block_size,)
            
            control_emotion = control_emotion[:block_size]
            pain_expressiveness = pain_expressiveness[:block_size]
            
            stimulus_abs = ctrl[2][batch_idx] # (block_size,)
            # past_latent = ctrl[3][batch_idx] # (block_size, dim_latent) #DONE: change this to the output of the model
            past_latent = torch.zeros((block_size,128), requires_grad=False, device=self.device) # (block_size, dim_latent)
            
            frames = []
            start = time.time()
            
            lenght = len(stimulus_abs)
            
            for frame_idx in range(1, lenght):
                
                # if block_size of control emotion and stimulus and past_latent is bigger than temoral
                
                # mask out the future
                des_mask = range(block_size-frame_idx-1,block_size)
                
                if block_size-frame_idx-1 < 0:
                    des_mask = range(0, block_size)

                zero_stimulus_abs = torch.zeros_like(pain_expressiveness, requires_grad=False, device=self.device)
            
                zero_stimulus_abs[des_mask] = stimulus_abs[frame_idx+1-block_size if frame_idx+1-block_size >= 0 else 0 : frame_idx+1]
                
                # add the batch dim
                _control_emotion, _pain_expressiveness, _zero_stimulus_abs, _zero_past_latent = map(lambda x: rearrange(x, "... -> 1 ...").contiguous(), [control_emotion, pain_expressiveness, zero_stimulus_abs, past_latent])
                
                new_ctrl = [_control_emotion, _pain_expressiveness, _zero_stimulus_abs, _zero_past_latent]
                
                # sample the frame
                frame = self(
                    num_imgs = 1,
                    ctrl = new_ctrl,
                    verbose=False,
                    guide = self.guide,
                    is_training = False
                )
                frame = reduce(frame, "batch time dim -> time dim", "mean")
                
                past_latent = torch.cat([past_latent, frame], dim=0)
                past_latent = past_latent[1:]
                
                frames.append(frame)
            
            # stack the video
            frames = torch.stack(frames)
            
            print("fps:", len(frames)/(time.time()-start))
            try:
                current_step = self.trainer.global_step if exists(self.trainer.global_step) else 0
            except:
                current_step = 0
            torch.save(frames, f"{self.sample_output_dir}/{current_step}_{batch_idx}.pt")
            torch.save(ctrl, f"{self.sample_output_dir}/{current_step}_{batch_idx}_ctrl.pt")
            
    @torch.no_grad() 
    def sample_imgs(self, ):
        # Collect the input shapes
        (x_0, ctrl) = self.val_outs
        
        # Produce 8 samples and log them
        imgs = self(
                num_imgs = x_0.shape[0],
                ctrl = ctrl,
                verbose = False,
                guide= self.guide # guide weight, should search for it
            )
        
        assert not torch.isnan(imgs).any(), 'NaNs detected in imgs!'
        
        # just dump the output tensor to disk then render later
        current_step = self.trainer.global_step
        
        torch.save(imgs, f"{self.sample_output_dir}/{current_step}.pt")
        
    def configure_optimizers(self) -> None:
        optim_conf = self.conf['OPTIMIZER']

        # Initialize the optimizer
        optim = AdamW(
            self.parameters(), 
            **optim_conf,   
        )
        return optim
        
        # self.warmup_iters = 5000

        # warmup = torch.optim.lr_scheduler.LinearLR(
        #     optim,
        #     start_factor=0.0001,
        #     end_factor=1,
        #     total_iters=self.warmup_iters,
        # )

        # red_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optim, factor=0.5, patience=3, min_lr=1e-6, verbose=True
        # )

        # lr_scheduler = {
        #     "scheduler": red_plateau,
        #     "interval": "epoch",
        #     "frequency": 2,
        #     "monitor": "val_loss",
        # }

        # return ([optim], [lr_scheduler, {"scheduler": warmup}])

    # * Functions that define what model actually predicts
    def c_skip(self, sigma: Optional[Tensor] = None) -> Tensor:
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: Optional[Tensor] = None) -> Tensor:
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma: Optional[Tensor] = None) -> Tensor:
        return (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma: Optional[Tensor] = None) -> Tensor:
        return torch.log(sigma.clamp(min=1e-20)) * 0.25

    # * Functions that define model training
    def loss_weight(self, sigma: Tensor) -> Tensor:
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def get_noise(self, batch_size: int) -> Tensor:
        eps = torch.randn((batch_size,), device=self.device)
        return (self.lognorm_mean + self.lognorm_std * eps).exp()

    # * Functions that define sampling strategy
    def get_timesteps(self, num_steps: int, rho: Optional[int] = None) -> Tensor:
        rho = default(rho, self.rho_schedule)

        inv_rho = 1 / rho

        tramp = torch.linspace(0, 1, num_steps, device=self.device)
        i_max = self.sigma_max**inv_rho
        i_min = self.sigma_min**inv_rho

        sigma = (i_max + tramp * (i_min - i_max)) ** rho

        return sigma

    def get_schedule(self, t: Tensor, **kwargs) -> Tensor:
        return t

    def get_scaling(self, t: Tensor, **kwargs) -> Tensor:
        return torch.ones_like(t)
