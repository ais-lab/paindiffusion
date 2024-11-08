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

# def generate_pyramid_scheduling_matrix(horizon: int, uncertainty_scale: float, sampling_timesteps: int) -> np.ndarray:
#     height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
#     scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
#     for m in range(height):
#         for t in range(horizon):
#             scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m

#     return torch.from_numpy(np.clip(scheduling_matrix, 0, sampling_timesteps))

# def full_sequence_scheduling_matrix(horizon: int, sampling_timesteps: int) -> np.ndarray:
#     return torch.from_numpy(np.arange(sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1))

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

        window_size = self.window_size
        context_size = self.context_size
        batch_size = x_0.shape[0]
        latent_dim = x_0.shape[-1]

        video_lenght = x_0.shape[1] # b t fs d

        use_x_c = default(use_x_c, self.self_cond)
        num_steps = default(num_steps, self.sample_steps)
        norm_undo = default(norm_undo, self.norm_backward)
        self.ode_solver = default(ode_solver, self.ode_solver)

        timestep = self.get_timesteps(num_steps)
        # timestep = timestep.flip(0)

        ctrl = self.ctrl_emb(ctrl, is_training=is_training) if exists(ctrl) else ctrl

        # TODO: implement rolling out mechanism by rolling diffusion
        # the model window size should be equal to the number of denoise steps

        # step 1: init schedule matrix
        # step 2: generate with init matrix
        # step 3: roll the rolling matrix

        # assert window_size == context_size + num_steps, "rolling diffusion need 1:1 ratio of window size and denoise"

        ini_sche = self.create_init_matrix(num_steps, context_size, timestep)
        win_sche = ini_sche[-1]
        end_sche = self.create_end_matrix(win_sche, noise_list=timestep)

        max_noise = max(timestep)

        placeholder = torch.rand((batch_size, video_lenght, self.frame_stack, latent_dim), device=self.device)
        placeholder = placeholder * max_noise

        placeholder = self.iterate_sche(
            ini_sche,
            placeholder,
            ctrl,
            guide,
            timesteps=timestep,
            window_size=window_size,
            context_size=context_size
        )

        placeholder = self.iterate_sche(
            win_sche, 
            placeholder, 
            ctrl, 
            guide, 
            current_idx=1, 
            stop_idx=video_lenght-window_size,
            timesteps=timestep,
            window_size=window_size,
            context_size=context_size
            )

        placeholder = self.iterate_sche(
            end_sche,
            placeholder,
            ctrl,
            guide,
            current_idx=video_lenght - (window_size),
            timesteps=timestep,
            window_size=window_size,
            context_size=context_size
        )

        return norm_undo(placeholder)

    def iterate_sche(self, sche, placeholder, ctrl, guide, current_idx=0, stop_idx=-1, timesteps=None, window_size=-1, context_size=-1):
        # TODO: implement the iterate sche function

        # init or ending -> no need to increement
        if len(sche.shape) == 2:

            region_of_interest = placeholder[:,current_idx:current_idx+sche.shape[-1]].clone()
            ctrl_of_interest = ctrl[:, :, current_idx:current_idx+sche.shape[-1]]
            for step, current_noise in enumerate(sche):
                if step == 0:
                    # past_noise = torch.zeros(current_noise.shape, device=self.device)
                    past_noise = torch.full_like(current_noise, min(timesteps), device=self.device)
                    next_noise = sche[step+1]

                elif step == len(sche) - 1:
                    past_noise = sche[step-1]
                    # next_noise = torch.zeros(current_noise.shape, device=self.device)
                    next_noise = torch.full_like(current_noise, min(timesteps), device=self.device)

                else:
                    past_noise = sche[step-1]
                    next_noise = sche[step+1]

                # print(current_idx + step)
                region_of_interest, _ = self.single_denoise_step_dpmpp(
                    region_of_interest,
                    (
                        past_noise,
                        current_noise,
                        next_noise
                    ),
                    ctrl=ctrl_of_interest,
                    guide=guide,
                    scaling=1
                )
                
                clean_frames = (current_noise == min(timesteps)).nonzero(as_tuple=True)[0]
                if len(clean_frames) == 0:
                    max_clean_frames = 0
                else:
                    max_clean_frames = max(clean_frames)
                
                placeholder[:, current_idx:sche.shape[-1]+current_idx] = region_of_interest[:,:]
            # print(clean_frames+current_idx)
            # placeholder[:, current_idx:sche.shape[-1]+current_idx] = region_of_interest

        # window -> increment idx
        elif len(sche.shape) == 1:

            max_noise = max(timesteps)
            min_noise = min(timesteps)

            # past is shift left by one and fill max
            # next is shift right by one and fill min

            past = torch.roll(sche, -1, 0)
            past[-1] = max_noise

            next = torch.roll(sche, 1, 0)
            next[0] = min_noise 
            
            clean_frames = (sche == min(timesteps)).nonzero(as_tuple=True)[0]
            max_clean_frames = max(clean_frames)

            for idx in range(current_idx, stop_idx):
                # print(idx , idx + sche.shape[-1])
                # roi : current_idx, current_idx + window_size

                region_of_interest = placeholder[:, idx : idx + sche.shape[-1]]
                ctrl_of_interest = ctrl[:, :, idx: idx + sche.shape[-1]]

                region_of_interest, _ = self.single_denoise_step_dpmpp(
                    region_of_interest,
                    (past, sche, next),
                    ctrl=ctrl_of_interest,
                    guide=guide,
                    scaling=1
                )

                placeholder[:, idx : idx + sche.shape[-1]] = region_of_interest[:,:]
                
                # print(max(clean_frames))
                # print(clean_frames+idx)

        return placeholder

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
        sig = self.get_noise(bs, t, batch_repeat=False, diffusion_rolling_noise=False)

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

        whole_loss = self.criterion(x_p, x_0, reduction='none')

        whole_loss : Tensor = reduce(whole_loss, 'b t ... -> b', 'mean').contiguous()

        loss =  whole_loss

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

        self.val_outs = batch.copy()

        x_0, ctrl, mask_weight = self.prepare_batch(batch)  

        # ctrl = self.ctrl_emb(ctrl) if exists(ctrl) else ctrl

        loss = self.compute_loss(x_0, ctrl = ctrl, is_training = False)

        self.log_dict({f'{namespace}_loss' : loss}, logger = True, on_step = True, sync_dist = True)

        # un stack the frame to get the original shape to support sampling later
        # x_0, ctrl = self.unstack(x_0, ctrl)

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

        # print(batch['x'].shape)
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
        
        print('number of forward:', self.count_forward)

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
    def get_noise(self, batch_size: int, time: int, batch_repeat=True, diffusion_rolling_noise=False) -> Tensor:

        if not diffusion_rolling_noise:
            if not batch_repeat:
                eps = torch.randn((batch_size, time,), device=self.device)
            else:
                eps_single_batch = torch.rand((time,), device=self.device)

                eps = repeat(eps_single_batch, 't -> b t', b=batch_size).contiguous()
            
        else:
            scale = torch.linspace(0, 1, time, device=self.device)
            init_matrix = self.create_init_matrix(time, self.context_size, scale)
            
            for i in range(batch_size):
                eps = torch.ones((time,), device=self.device)
                # init window
                if random() < 0.5:
                    
                    random_idx = torch.randint(0, len(init_matrix), (1,)).item()
                    
                    eps = eps * init_matrix[random_idx]
                
                # rolling window
                else:
                    
                    eps = eps * init_matrix[-1]
                
            

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

    # def get_schedule(self, t: Tensor, window_size: int=16, context_size: int=4, clean_horizon_size: int=2, mode: str="elbow", **kwargs) -> Tensor:

    def create_init_matrix(self, num_denoise_steps, clean_frames, noise_list):
        num_init_steps = num_denoise_steps + clean_frames
        init_matrix = torch.full((num_init_steps, num_init_steps), noise_list.max(), device=self.device)    

        for row_id in range(num_init_steps):
            if row_id < clean_frames:
                fill_zeros = torch.full((clean_frames - row_id,), noise_list.min(), device=self.device)
                fill_matrix = torch.cat((noise_list[:num_denoise_steps], fill_zeros), dim=0)
                init_matrix[row_id, row_id:] = fill_matrix
            else:
                init_matrix[row_id, row_id:] = noise_list[:num_denoise_steps-row_id+clean_frames]

        return init_matrix.T

    def create_end_matrix(self, window_sche, noise_list):
        window_size = len(window_sche)
        # end_matrix = torch.zeros((window_size, window_size), device=self.device)
        end_matrix = torch.full((window_size, window_size), min(noise_list), device=self.device)

        for row_id in range(window_size):    
            end_matrix[row_id, row_id:] = window_sche[:window_size-row_id]

        return end_matrix

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
