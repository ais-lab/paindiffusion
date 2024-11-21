import os
from random import random
import time
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm
from .diffusion import Diffusion

from .module.utils.misc import exists, groupwise
from .module.utils.misc import default
from .module.utils.misc import enlarge_as

from typing import Callable, Dict, Optional, Tuple, Union
from einops import rearrange, reduce, repeat

from math import sqrt, log, expm1


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
        
    def forward(self, ctrl, is_training=False):

        # [B, sequence_len], [B, sequence_len], [B, sequence_len]
        if len(ctrl) == 4:
            control_emotion, pain_expressiveness, stimulus_abs, _ = ctrl
        elif len(ctrl) == 3:
            control_emotion, pain_expressiveness, stimulus_abs = ctrl
        
        # DONE: normalize the condition signal with the data mean and std 
        control_emotion = (control_emotion - self.conf['emotion']['mean']) / self.conf['emotion']['std']
        pain_expressiveness = (pain_expressiveness - self.conf['expressiveness']['mean']) / self.conf['expressiveness']['std']
        stimulus_abs = (stimulus_abs - self.conf['stimulus_abs']['mean']) / self.conf['stimulus_abs']['std']

        control_emotion, pain_expressiveness, stimulus_abs = map(
            lambda x: rearrange(x, "b ... -> b ... 1").contiguous(),
            [control_emotion, pain_expressiveness, stimulus_abs],
        )       
        
        # fill 0 to the past_emb to match the sequence length
        
        if is_training and random() < 0.3: # crop at random length
            random_length = torch.randint(1, stimulus_abs.shape[1], (1,)).item()
            stimulus_abs = stimulus_abs[:, :random_length]
            
        control_emotion = self.emotion(control_emotion)
        pain_expressiveness = self.pain_expressive(pain_expressiveness)
        stimulus_abs = self.stimulus_abs(stimulus_abs) 
        
        fill_sequence_stimulus = control_emotion.shape[1] - stimulus_abs.shape[1]
        stimulus_abs = F.pad(stimulus_abs, (0,0,0,0,fill_sequence_stimulus,0), "constant", 0)
        
        # target shape: [B, sequence_lenth, emb_dim * num_ctrl]
        # out = torch.cat([control_emotion, pain_expressiveness, stimulus_abs, past_latent], dim=-1)
        out = torch.stack([control_emotion, pain_expressiveness, stimulus_abs]) # [num_ctrl, B, sequence_lenth, emb_dim]
        
        return out

# DONE: implement different noise level for each frame (diffusion forcing) -> _generate_noise_level function
#       1. generate noise level for each frame in the get_noise function
# TODO: rolling sampling video
#       1. sample a long video and then call eval after training for each 50k steps
# DONE: Sample the output at each val end
# TODO: Sample the output for evaluation in the end

logsnr = lambda sig : -torch.log(sig)

def generate_pyramid_scheduling_matrix(horizon: int, uncertainty_scale: float, sampling_timesteps: int) -> np.ndarray:
    height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
    scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
    for m in range(height):
        for t in range(horizon):
            scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m

    return torch.from_numpy(np.clip(scheduling_matrix, 0, sampling_timesteps))

def full_sequence_scheduling_matrix(horizon: int, sampling_timesteps: int) -> np.ndarray:
    return torch.from_numpy(np.arange(sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1))

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
        warmup_steps: int = 1000,
        frame_stack: int = 1,
        window_size: int = 16,
        context_size: int = 4,
        uncertainty_scale: float = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            ode_solver=ode_solver,
            **kwargs,
        )

        self.warmup_steps = warmup_steps
        self.frame_stack = frame_stack

        self.window_size = window_size
        print(f"window size: {window_size}")
        self.context_size = context_size
        print(f"context size: {context_size}")
        self.uncertainty_scale = uncertainty_scale
        print(f"uncertainty scale: {uncertainty_scale}")

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
        self.norm_backward = lambda x: 0.5 * (1 + x)

        self.ctrl_emb: Optional[nn.Module] = ControlEmbedding(
            emb_dim=128, in_dim=1, conf=self.conf['DATA_STATS']
        )

        self.sample_output_dir = sample_output_dir

        self.save_hyperparameters(ignore=['model'])

        self.collect_xs = []

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

        num_controll = 3
        batch_size = ctrl.shape[1]

        assert len(guide) == num_controll

        original_ctrl = ctrl.clone()

        # cond = self.predict(*args, ctrl=ctrl, **kwargs)
        sum_scale = sum(guide)

        # bring the controll signal to the front to iterate over
        # _ctrl = rearrange(original_ctrl, "b l (d c) -> c b l d", c=num_controll)

        # zero out the control signal
        null_ctrl = [rearrange(original_ctrl.clone(), "c b l fs d -> b l (fs c) d").contiguous()]
        for idx, control_signal in enumerate(original_ctrl):
            _null_ctrl = self.null_ctrl(control_signal)

            with_null_ctrl = original_ctrl.clone()
            with_null_ctrl[idx] = _null_ctrl

            # bring the controll signal back to the back to pass to the model
            with_null_ctrl = rearrange(with_null_ctrl, "c b l fs d -> b l (fs c) d").contiguous()

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
        current_denoise_step: int = 0,
        schedule: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply the backbone model to come up with a prediction, the
        nature of which depends on the diffusion objective (can either
        be noise|x_start|v prediction).
        """

        bs, *_, device = x_t.shape, x_t.device

        if isinstance(sig, float):
            sig = torch.full((bs,), sig, device=device)

            # add batch to noise
            # if len(sig.shape) == 1:
            #     sig = repeat(sig, 't -> b t', b = x_t.shape[0])

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

        # if clamp:
        #     out = out.clamp(-1.0, 1.0)

        return out

    @torch.no_grad()
    def forward(
        self,
        x_0,
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

        # batch = {
        #     'x': x_0,
        #     'ctrl': ctrl
        # }

        # x_0, ctrl, _ = self.prepare_batch(batch)

        latent_dim = 128
        batch_size = num_imgs
        window_size = self.window_size
        context_size = self.context_size
        uncertainty_scale = self.uncertainty_scale

        chunk_size = window_size - context_size # how many new time step should be generated
        # clean_horizon_size = 1
        # schedule_mode = "elbow"
        video_lenght = x_0.shape[1] # b t fs d

        use_x_c = default(use_x_c, self.self_cond)
        num_steps = default(num_steps, self.sample_steps)
        norm_undo = default(norm_undo, self.norm_backward)
        self.ode_solver = default(ode_solver, self.ode_solver)

        # window_size: int=16, context_size: int=4, clean_horizon_size: int=2
        timestep = self.get_timesteps(num_steps)
        timestep = timestep.flip(0)

        # Encode the condition using the sequence encode

        ctrl = self.ctrl_emb(ctrl, is_training=is_training) if exists(ctrl) else ctrl

        max_noise = torch.index_select(timestep, 0, torch.tensor(num_steps - 1, device=self.device))

        curr_frame = 0
        xs_pred = None

        xs_pred = self.norm_forward(x_0[:,:context_size].clone())
        curr_frame += context_size

        forward_count = 0

        while curr_frame < video_lenght:

            # log_xs_pred = []

            # xs_pred_backup = xs_pred.clone()
            # # multiple sampling to draw diagram
            for i in range(1):

                #     xs_pred = xs_pred_backup.clone()

                xs_pred_this_try = []

                if chunk_size > 0:
                    horizon = min(video_lenght - curr_frame, chunk_size)
                else:
                    horizon = video_lenght - curr_frame
                assert horizon <= window_size, "horizon should be smaller than window size of the denoise model"

                scheduling_matrix = generate_pyramid_scheduling_matrix(horizon, uncertainty_scale=uncertainty_scale, sampling_timesteps=len(timestep)-1).to(self.device)
                # TODO: remember to roll back to the original scheduling matrix
                # scheduling_matrix = full_sequence_scheduling_matrix(horizon, sampling_timesteps=len(timestep)-1).to(self.device)

                xs_pred = self.single_chunk_prediction(ctrl, guide, latent_dim, batch_size, window_size, timestep, max_noise, curr_frame, horizon, scheduling_matrix, xs_pred)

                #     xs_pred_this_try.append(xs_pred[:,start_frame:].clone())
                # self.collect_xs.append(xs_pred_this_try)

            # torch.save(log_xs_pred, f"sample_distribution_mapping.pt")

            curr_frame += horizon 

        return norm_undo(xs_pred)

    def single_chunk_prediction(self, ctrl, guide, latent_dim, batch_size, window_size, timestep, max_noise, curr_frame, horizon, scheduling_matrix, xs_pred=None,):

        chunk = torch.randn((batch_size, horizon, self.frame_stack, latent_dim), 
                                    device=self.device, 
                                    # dtype=torch.float64
                                    )
        chunk = chunk * max_noise # this to follow dpmpp to scale noise to largest noise level 

        if exists(xs_pred):
            xs_pred = torch.cat([xs_pred, chunk], dim=1) # cat on time dim
        else:
            # TODO: solve this to have no context
            xs_pred = chunk

        start_frame = max(0, curr_frame + horizon - window_size)
        
        # print(f"curr_frame: {curr_frame}, horizon: {horizon}, window_size: {window_size}, start_frame: {start_frame}")
        # print(f"xs_pred shape: {xs_pred.shape}")

        for m in range(scheduling_matrix.shape[0]-1):
            # add noise level for the context frame as zero as the diffusion forcing proposed  context->[0,0, a,b,c]<-chunk from schedule
            if m == 0:
                past_noise = torch.zeros(scheduling_matrix[m].shape, device=self.device) 
            else:
                past_noise = scheduling_matrix[m-1]

            current_noise = scheduling_matrix[m]

            if m == scheduling_matrix.shape[0]-2:
                next_noise = torch.zeros(scheduling_matrix[m].shape, device=self.device)
            else:
                next_noise = scheduling_matrix[m+1]

            past_noise, current_noise, next_noise = map(
                        lambda x: torch.cat([torch.zeros((curr_frame,), device=self.device), x], dim=0),
                        (past_noise, current_noise, next_noise)
                    )

            past_noise, current_noise, next_noise = map(
                        lambda x: torch.index_select(timestep, 0, x.int()),
                        (past_noise, current_noise, next_noise),
                    )
            
            xs_pred[:,start_frame:], _ = self.single_denoise_step_dpmpp(
                        xs_pred[:,start_frame:],
                        (
                            past_noise[start_frame:],
                            current_noise[start_frame:],
                            next_noise[start_frame:],
                        ),
                        ctrl=ctrl[:,:, start_frame : curr_frame + horizon], # num_ctrl, batch, lenght, fs, dim
                        guide=guide,
                        scaling=1.
                    )

        return xs_pred

    def sample_a_chunk(self, ctrl, guide, past_frames=None, scheduling_matrix=None):
        latent_dim = 128
        batch_size = 1
        window_size = self.window_size
        context_size = self.context_size
        uncertainty_scale = self.uncertainty_scale

        chunk_size = window_size - context_size # how many new time step should be generated
        horizon = chunk_size if past_frames is not None else window_size
        
        num_steps = self.sample_steps
        norm_undo = self.norm_backward

        # window_size: int=16, context_size: int=4, clean_horizon_size: int=2
        timestep = self.get_timesteps(num_steps)
        timestep = timestep.flip(0)
        max_noise = torch.index_select(timestep, 0, torch.tensor(num_steps - 1, device=self.device))
        
        scheduling_matrix = generate_pyramid_scheduling_matrix(horizon, uncertainty_scale=4, sampling_timesteps=len(timestep)-1).to(self.device)
        # scheduling_matrix = full_sequence_scheduling_matrix(horizon, sampling_timesteps=len(timestep)-1).to(self.device)

        for idx, cond in enumerate(ctrl):
            if len(cond.shape) != 2:
                continue
            
            # print(cond.shape)
            ctrl[idx] = rearrange(cond, "b (t fs) -> b t fs", fs=self.frame_stack).contiguous()

        ctrl = self.ctrl_emb(ctrl, is_training=False) if exists(ctrl) else ctrl
        
        if exists(past_frames):
            past_frames = rearrange(past_frames, "b (t fs) d -> b t fs d", fs=self.frame_stack).contiguous()
            past_frames[..., :3] *= 100 # scale the jaw pose back to the original scale
            xs_pred = self.norm_forward(past_frames)
            xs_pred = xs_pred[:, -context_size:]
        else:
            xs_pred = None

        imgs = self.single_chunk_prediction(
            ctrl=ctrl,
            guide=guide,
            latent_dim=latent_dim,
            batch_size=batch_size,
            window_size=window_size,
            timestep=timestep,
            max_noise=max_noise,
            curr_frame=0 if not exists(past_frames) else context_size,
            horizon=horizon,
            scheduling_matrix=scheduling_matrix,
            xs_pred=xs_pred,
        )
        
        imgs = norm_undo(imgs)
        
        imgs[..., :3] /= 100
        
        if exists(past_frames):
            imgs = imgs[:, context_size:]

        imgs = rearrange(imgs, "b t fs d -> b (t fs) d").contiguous()

        return imgs

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

        # Encode the condition using the sequence encoder
        ctrl = self.ctrl_emb(ctrl, is_training) if exists(ctrl) else ctrl
        ctrl = rearrange(ctrl, "c b l fs d -> b l (c fs) d").contiguous() if exists(ctrl) else ctrl

        # Normalize input images
        x_0 = norm_fn(x_0)

        bs, t, *_ = x_0.shape

        # Get the noise and scaling schedules
        sig = self.get_noise(bs, t)

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

        # prepare the objective

        # DONE: change to v-pred | confirm this change tomorrow => this is not correct
        # v_pred = enlarge_as(sig, x_0) * eps - enlarge_as(sig, x_0) * x_0
        # loss (v, x_p)

        # split into 2 loss term as jaw pose feature are very small compare to expression, that make the model only optimize pose, the trick is the same with my last paper
        # x_p_jaw = x_p[..., :3]
        # x_0_jaw = x_0[..., :3]

        # x_p_exp = x_p[..., 3:103]
        # x_0_exp = x_0[..., 3:103]

        # Compute the reconstruction loss
        # loss_jaw = self.criterion(x_p_jaw, x_0_jaw, reduction = 'none')
        # loss_exp = self.criterion(x_p_exp, x_0_exp, reduction = 'none')
        whole_loss = self.criterion(x_p, x_0, reduction='none')

        # loss_jaw : Tensor = reduce(loss_jaw, 'b t ... -> b', 'mean').contiguous()
        # loss_exp : Tensor = reduce(loss_exp, 'b t ... -> b', 'mean').contiguous()

        # having this as mean to the batch work better
        whole_loss : Tensor = reduce(whole_loss, 'b t ... -> b', 'mean').contiguous()

        alpha = 0.8

        loss = (
            # loss_jaw * (1 - alpha)
            # + loss_exp * alpha
            0
            + whole_loss
        )

        # Add loss weight
        sig = sig.mean() # todo check this to have different mean: confirm this work better
        loss *= self.loss_weight(sig)
        return loss.mean()

    # DONE: implement frame stack (move the frame to channel to make it faster), stack the condition as well
    #       1. stacking to lower the time b (fs t) c d -> b t (fs c) d
    #       2. unstacking to get the original shape b t (fs c) d -> b (fs t) c d
    #       3. reweight the loss as in difusion forcing paper ()
    # in the original paper, they use 4 fs for video prediction, 8 for mineccraft and 10 for maze planning (-> higher fs for less complex task)

    def prepare_batch(self, batch : Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_0  = batch[self.data_key]
        ctrl = batch[self.ctrl_key] if exists(self.ctrl_key) else None

        frame_stack = self.frame_stack

        batch_size, num_frames, *_ = x_0.shape

        x_0 = rearrange(x_0, "b (t fs) d -> b t fs d", fs=frame_stack).contiguous()

        if exists(ctrl):
            for idx, cond in enumerate(ctrl):

                if len(cond.shape) != 2:
                    continue

                ctrl[idx] = rearrange(cond, "b (t fs) -> b t fs", fs=frame_stack).contiguous()

        mask_weight = torch.ones(batch_size, num_frames).to(x_0.device)

        return x_0, ctrl, mask_weight

    def unstack(self, x_0, ctrl) -> Tensor:
        x_0 = rearrange(x_0, "b t fs d -> b (t fs) d", fs=self.frame_stack).contiguous()     
        ctrl = [rearrange(cond, "b t fs -> b (t fs)").contiguous() for cond in ctrl] if exists(ctrl) else None
        return x_0, ctrl

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ) -> None:
        # OLDFIXME which should come first?
        # manually warm up lr without a scheduler

        # DONE: change load the parameter from the input of the function
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.conf['OPTIMIZER']['lr']

        # update params
        optimizer.step(closure=optimizer_closure)

    # * Lightning Module functions
    def training_step(self, batch : Dict[str, Tensor], batch_idx : int) -> Tensor:
        # Extract the starting images from data batch

        x_0, ctrl, mask_weight = self.prepare_batch(batch)

        # DONE: random drop the control signal
        num_controll = 3
        if isinstance(self.drop_probs, float):
            self.drop_probs = [self.drop_probs] * num_controll
        else:
            assert len(self.drop_probs) == num_controll, "The length of drop_prob should be equal to the number of control signal"

        if len(ctrl) == 4: # control_emotion, pain_expressiveness, stimulus_abs, pspi_no_au43 (pspi was added for evaluation)
            # drop the pspi_no_au43
            ctrl = ctrl[:-1]

        for idx, control_signal in enumerate(ctrl):
            if random() < self.drop_probs[idx]:
                ctrl[idx] = self.null_ctrl(control_signal)

        loss = self.compute_loss(x_0, ctrl = ctrl, is_training = True)

        self.log_dict({'train_loss' : loss}, logger = True, on_step = True, sync_dist = True)

        return loss

    def validation_step(self, batch : Dict[str, Tensor], batch_idx : int, namespace="val") -> Tensor:
        # Extract the starting images from data batch

        x_0, ctrl, mask_weight = self.prepare_batch(batch)  

        # ctrl = self.ctrl_emb(ctrl) if exists(ctrl) else ctrl

        loss = self.compute_loss(x_0, ctrl = ctrl, is_training = False)

        self.log_dict({f'{namespace}_loss' : loss}, logger = True, on_step = True, sync_dist = True)

        # un stack the frame to get the original shape to support sampling later
        # x_0, ctrl = self.unstack(x_0, ctrl)

        self.val_outs = batch

        return x_0, ctrl

    def test_step(self, batch, batch_idx):

        if batch_idx > 200:

            if not os.path.exists("sample_distribution_mapping.pt"):            
                torch.save(self.collect_xs, f"sample_distribution_mapping.pt")
            pass
        else:
            self.sample_imgs(batch, namespace="test", batch_idx=batch_idx)

    @torch.no_grad()
    def on_validation_epoch_end(self, 
                                # val_outs : Tuple[Tensor, ...]
                                ) -> None:
        '''
            At the end of the validation cycle, we inspect how the denoising
            procedure is doing by sampling novel images from the learn distribution.
        '''

        # DONE: sample a video instead of an image and call the renderer

        # done: investigate the bottleneck of the validation sample
        # maybe it comes with the inference engine of the lightning module?
        # run a profiller to figure out the bottleneck
        # if not able to fix it then just sample a bunch of frames instead of sampling one by one

        batch = self.val_outs
        self.sample_imgs(batch)

        # os.system('/home/tien/miniconda3/envs/work39_torch2/bin/python /home/tien/inferno/render_from_exp.py --input_path "/media/tien/SSD-NOT-OS/pain_intermediate_data/output_video/10_32dim_lossweight_ema_splitloss/sample/*_with_ctrl.pt" --output_dir "/media/tien/SSD-NOT-OS/pain_intermediate_data/output_video/10_32dim_lossweight_ema_splitloss/sample/" --video_render true')

    @torch.no_grad() 
    def sample_imgs(self, batch, namespace='val', batch_idx=0, save=True):

        print(batch['x'].shape)

        self.count_forward = 0

        x_0, ctrl, *_ = self.prepare_batch(batch)

        # Produce 8 samples and log them
        imgs = self(
                x_0,
                num_imgs = x_0.shape[0],
                ctrl = ctrl,
                guide= self.guide # guide weight, should search for it
            )

        assert not torch.isnan(imgs).any(), 'NaNs detected in imgs!'

        # just dump the output tensor to disk then render later
        try:
            current_step = self.trainer.global_step
        except:
            current_step = 0

        # unscale jaw pose refer to utils/biovid.py latent jaw pose that scale it by multiply 100
        # so we need to scale it back to the original scale
        imgs[..., :3] /= 100

        imgs = rearrange(imgs, "b t fs d -> b (t fs) d").contiguous()

        saved_objects = {
            'x' : imgs,
            # 'current_step' : current_step,
            # 'batch_idx': batch_idx,
            'ctrl' : ctrl,
            # "start_frame_id":batch['start_frame_id'],
            # "end_frame_id":batch['end_frame_id'],
            # 'video_name':batch['video_name'],
        }

        print("num forward:", self.count_forward)

        if save:
            torch.save(saved_objects, f"{self.sample_output_dir}/{current_step}_{batch_idx}_{namespace}_with_ctrl.pt")
            print(f"Saved the output at {self.sample_output_dir}/{current_step}_{batch_idx}_{namespace}_with_ctrl.pt")

        return saved_objects

    def configure_optimizers(self) -> None:
        optim_conf = self.conf['OPTIMIZER']

        red_plateau_conf: Dict = optim_conf.pop('red_plateau')

        # Initialize the optimizer
        optim = AdamW(
            self.parameters(), 
            **optim_conf,   
        )

        red_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=red_plateau_conf.get('factor', 0.5), patience=red_plateau_conf.get('patience', 3), min_lr=red_plateau_conf.get('min_lr', 4e-5), verbose=True
        )

        lr_scheduler = {
            "scheduler": red_plateau,
            "interval": red_plateau_conf.get('interval', 'epoch'),
            "frequency": red_plateau_conf.get('frequency', 1),
            "monitor": red_plateau_conf.get('monitor', 'val_loss'),
        }

        return ([optim], [lr_scheduler])

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

    # DONE: change this for diffusion forcing
    def get_noise(self, batch_size: int, time: int) -> Tensor:
        eps = torch.randn((batch_size, time,), device=self.device)
        return (self.lognorm_mean + self.lognorm_std * eps).exp()

    # * Functions that define sampling strategy
    # DONE: change this to have the time step for each frame
    def get_timesteps(self, num_steps: int, rho: Optional[int] = None) -> Tensor:
        rho = default(rho, self.rho_schedule)

        inv_rho = 1 / rho

        tramp = torch.linspace(0, 1, num_steps, device=self.device)
        i_max = self.sigma_max**inv_rho
        i_min = self.sigma_min**inv_rho

        sigma = (i_max + tramp * (i_min - i_max)) ** rho

        return sigma

    @staticmethod
    def get_delta_denoise(window_size: int=16, context_size: int=4, clean_horizon_size: int=2, max_steps:int=35):
        denoise_horizon = window_size - clean_horizon_size - context_size
        delta_denoise = max_steps / denoise_horizon
        return delta_denoise

    def get_schedule(self, t: Tensor, window_size: int=16, context_size: int=4, clean_horizon_size: int=2, mode: str="elbow", **kwargs) -> Tensor:

        # DONE: generate the matrix for the noise level for each frame

        if mode == "same":
            return repeat(t, 't -> t w', w = window_size).contiguous()

        if mode == "pyramid":
            return generate_pyramid_scheduling_matrix(t.shape[0], uncertainty_scale=0.5, sampling_timesteps=window_size)

        if mode == "elbow":
            max_steps = t.shape[0]

            # starting
            delta_denoise = self.get_delta_denoise(window_size, context_size, clean_horizon_size, max_steps)

            noise_list = [[max_steps-1 for i in range(window_size)]]
            current_clean = 0
            while True:

                # if current_clean == context_size:
                #     break

                if noise_list[-1][-1] == 0:
                    break

                noise_level = [max_steps-1 for i in range(window_size)]

                clean_noise_level = noise_list[-1][current_clean:current_clean+clean_horizon_size][0] - delta_denoise

                if clean_noise_level < 0:
                    clean_noise_level = 0

                noise_level[current_clean:current_clean+clean_horizon_size] = [int(clean_noise_level)]*clean_horizon_size

                for i in range(0, current_clean):
                    noise_level[i] = 1

                count = 1
                for i in range(current_clean+clean_horizon_size, len(noise_level)):
                    noise = int(clean_noise_level + count*delta_denoise)
                    if noise >= max_steps:
                        noise = max_steps - 1
                    noise_level[i] = noise
                    count += 1

                noise_list.append(noise_level)

                if 0 in noise_list[-1]:
                    current_clean += 1

            # to generate the final frame
            noise_list.append(noise_level)

            return noise_list

    def get_scaling(self, t: Tensor, **kwargs) -> Tensor:
        return torch.ones_like(t)

    def single_denoise_step_dpmpp(self,
        x_t : Tensor,
        past_current_next_noise : Tuple[Tensor],
        scaling  : Tensor,
        ctrl  : Optional[Tensor] = None,
        clamp : bool = False,
        guide : Union[float, list]=1.,
        x_c : Optional[Tensor] = None,
        ):

        self.count_forward = self.count_forward + 1

        sigm1, sig, sigp1 = past_current_next_noise # past, current, next noise level

        s = scaling

        # add batch to noise
        sigm1, sig, sigp1 = map(lambda x: repeat(x, 't -> b t', b = x_t.shape[0]).contiguous(), (sigm1, sig, sigp1))

        p_t = self.follow(x_t, sig, x_c = x_c if exists(x_c) else None, ctrl = ctrl, guide = guide, clamp = clamp)

        sigm1, sig, sigp1 = map(lambda x: enlarge_as(x, x_t), (sigm1, sig, sigp1))

        l_t, l_tp1 = logsnr(sig), logsnr(sigp1)
        h_tp1 : float = l_tp1 - l_t

        if x_c is None:
            # if x_c is None or sigp1 == 0:
            dxdt = p_t
        else:
            epsilon = 1e-7 # to avoid nan
            h_t = l_t - logsnr(sigm1)
            r_t = h_t / (h_tp1 + epsilon)

            dxdt = (1 + 1 / (2 * r_t + epsilon)) * p_t - (1 / (2 * r_t + epsilon)) * x_c

        x_t = sigp1 / sig * x_t - s * torch.expm1(-h_tp1) * dxdt
        x_c = p_t

        return x_t, x_c

    def dpmpp(
        self,
        shape : Tuple[int,...],
        schedule : Tensor,
        scaling  : Tensor,
        ctrl  : Optional[Tensor] = None,
        use_x_c : Optional[bool] = None,
        clamp : bool = False,
        guide : Union[float, list]=1.,
        verbose : bool = False,
    ) -> Tensor:
        '''
            DPM++ Solver (2° order - 2M variant) from:
            https://arxiv.org/pdf/2211.01095 (Algorithm 2)
        '''

        use_x_c = default(use_x_c, self.self_cond)

        N = len(schedule)

        x_c = None # Parameter for self-conditioning
        x_t = schedule[0] * torch.randn(shape, device = self.device)

        # Iterate through the schedule|scaling three at a time (sigm1, sig, sigp1) being the past noise level, current noise level and the next noise level
        pars = zip(groupwise(schedule, n = 3, pad = -1, extend = False), scaling)

        for (sigm1, sig, sigp1), s in tqdm(pars, total = N, desc = 'DPM++', disable = not verbose):

            x_t, x_c = self.single_denoise_step_dpmpp(
                x_t,
                (sigm1, sig, sigp1),
                s,
                ctrl=ctrl,
                clamp=clamp,
                guide=guide,
                x_c=x_c if use_x_c else None,
            )

        return x_t
