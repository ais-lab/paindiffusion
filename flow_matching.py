from random import random, choice
import numpy as np
import tyro
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import yaml
from einops import rearrange, repeat
from lightning import Trainer 
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from diffusion.module.utils.biovid import exists, BioVidDM
from diffusion.module.temporal_latent_unet import Unet1D as VideoLatentUnet
from diffusion.module.utils.ema import EMA
from diffusion.module.control_embedding import ControlEmbedding
import os

from diffusion.module.utils.misc import enlarge_as
from diffusion.module.utils.biovid import bilateral_filter

@dataclass
class TrainingConfig:
    config_path: str = '/home/tien/paindiffusion/configure/flow_matching.yml'
    sample_output_dir: str = 'samples'
    max_epochs: int = 30
    learning_rate: float = 1e-4
    drop_probs: list = field(default_factory=lambda: [0.3, 0.3, 0.3])
    batch_size: int = 32
    num_workers: int = 4
    frame_stack: int = 4
    devices: int = 1
    train: bool = True
    validate: bool = True
    test: bool = False
    load_from_checkpoint: bool = False
    checkpoint_path: Optional[str] = None
    fast_check: bool = False
    use_logger: bool = True
    
def generate_pyramid_scheduling_matrix(horizon: int, uncertainty_scale: float, sampling_timesteps: int) -> np.ndarray:
    height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
    scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
    for m in range(height):
        for t in range(horizon):
            scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m

    return torch.from_numpy(np.clip(scheduling_matrix, 0, sampling_timesteps))

class FlowMatchingModule(LightningModule):
    def __init__(self, net, ctrl_emb, lr=0.001, frame_stack=4, drop_probs=[0.3, 0.3, 0.3], sample_output_dir='samples', guide=[1, 1, 1]):
        super().__init__()
        self.net:VideoLatentUnet = net
        self.ctrl_emb:ControlEmbedding = ctrl_emb
        self.lr = lr
        self.frame_stack = frame_stack
        self.drop_probs = drop_probs
        self.criterion = nn.MSELoss(reduction='mean')
        self.sample_output_dir = sample_output_dir
        self.val_outs = None
        self.guide = guide

    def forward(self, x, t, ctrl=None, cfg=False):
        if ctrl is not None:
            
            if self.training:
                ctrl = self.ctrl_emb.dropout_ctrl(ctrl, self.drop_probs)
            
            ctrl = self.ctrl_emb(ctrl, is_training=self.training)
            
        if not cfg:
            ctrl = rearrange(ctrl, "c b l fs d -> b l (c fs) d").contiguous() if exists(ctrl) else ctrl
            return self.net(x, t, ctrl=ctrl)
        else:
            return self.net.classifier_free_guidance(x, t, ctrl=ctrl, guide=self.guide)
    
    def prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_0 = batch['x']
        ctrl = batch['ctrl']
        
        batch_size, num_frames, *_ = x_0.shape
        x_0 = rearrange(x_0, "b (t fs) d -> b t fs d", fs=self.frame_stack).contiguous()

        if exists(ctrl):
            for idx, cond in enumerate(ctrl):
                if len(cond.shape) != 2:
                    continue
                ctrl[idx] = rearrange(cond, "b (t fs) -> b t fs", fs=self.frame_stack).contiguous()

        mask_weight = torch.ones(batch_size, num_frames).to(x_0.device)
        return x_0, ctrl, mask_weight

    def training_step(self, batch, batch_idx):
        x_1, ctrl, mask_weight = self.prepare_batch(batch)
        t = torch.rand((x_1.shape[0], x_1.shape[1]), device=self.device)
        # t = t.unsqueeze(-1).repeat(1, x_1.shape[1])
        x_0 = torch.randn_like(x_1)
        t_expanded = enlarge_as(t, x_1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        dx_t = x_1 - x_0
        pred = self(x_t, t, ctrl=ctrl)
        loss = self.criterion(pred, dx_t)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_1, ctrl, mask_weight = self.prepare_batch(batch)
        t = torch.rand((x_1.shape[0], x_1.shape[1]), device=self.device)
        # t = t.unsqueeze(-1).repeat(1, x_1.shape[1])
        x_0 = torch.randn_like(x_1)
        t_expanded = enlarge_as(t, x_1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        dx_t = x_1 - x_0
        pred = self(x_t, t, ctrl=ctrl)
        loss = self.criterion(pred, dx_t)
        self.log('val_loss', loss)
        self.val_outs = batch
        return loss
       
    def flow_step(self, x_t, t_start, t_end, ctrl):
        
        # t_start = enlarge_as(t_start, x_t)
        dt = t_end - t_start
        half_dt = dt / 2
        
        # First inner step
        v1 = self(x_t, t_start, ctrl=ctrl, cfg=True)
        x_mid = x_t + v1 * enlarge_as(half_dt, v1)
        
        # Second step from midpoint
        v2 = self(x_mid, t_start + half_dt, ctrl=ctrl, cfg=True)
        
        # Final position
        return x_t + enlarge_as(dt, v2) * v2
    
    def sample_imgs(self, batch, save=False, n_steps=4):
        x_0, ctrl, mask_weight = self.prepare_batch(batch)
        
        window_size = 16
        context = 4

        time_steps = torch.linspace(0, 1.0, n_steps+1, device=self.device)
        
        video_length = x_0.shape[1]
        curr_frame = 0
        chunk_size = window_size - context
        
        x_preds = None
        
        while curr_frame < video_length:

            if chunk_size > 0:
                horizon = min(chunk_size, video_length - curr_frame)
            else:
                horizon = video_length - curr_frame
            
            with_context = curr_frame >= context
            
            window_scheduling_matrix = generate_pyramid_scheduling_matrix(
                horizon=horizon,
                uncertainty_scale=1,
                sampling_timesteps=n_steps
            )
            
            # flip the scheduling matrix
            window_scheduling_matrix = torch.flip(window_scheduling_matrix, [0, 1])
            
            start_frame = max(0, curr_frame + horizon - window_size)
            end_frame = min(video_length, curr_frame + horizon)
                        
            window_x = torch.randn(
                (x_0.shape[0], horizon, self.frame_stack, 128), 
                device=self.device,
                )
            
            if exists(x_preds):
                window_x = torch.cat((x_preds[:,start_frame:], window_x), dim=1)
            
            window_ctrl = [ctrl[i][:, start_frame:end_frame] for i in range(len(ctrl))] if exists(ctrl) else None

            current_context_size = end_frame - start_frame - horizon
        
            window_x = self.sample_a_chunk(window_scheduling_matrix, window_x, window_ctrl, time_steps, with_context, context_size=current_context_size)
        
            x_preds = torch.cat((x_preds, window_x[:, -horizon:]), dim=1) if exists(x_preds) else window_x
            
            curr_frame += horizon 
        
        x_preds[..., :3] /= 100
        
        x_preds = rearrange(x_preds, "b t fs d -> b (t fs) d")

        saved_object = {
            'x' : x_preds,
            'ctrl' : ctrl,
            "start_frame_id":batch['start_frame_id'],
            "end_frame_id":batch['end_frame_id'],
            'video_name':batch['video_name'],
        }
        
        try:
            current_step = self.trainer.global_step
        except:
            current_step = 0
        
        if save:
            torch.save(saved_object, f"{self.sample_output_dir}/{current_step}_with_ctrl.pt")
            
        return saved_object

    def sample_a_chunk(self, scheduling_matrix, window_x, window_ctrl, time_steps, with_context, context_size):
        
        sigma = [1e-4, 2e-4]
        
        for m in range(scheduling_matrix.shape[0]-1):

            current_noise = scheduling_matrix[m]

            if m == scheduling_matrix.shape[0]-1:
                next_noise = torch.ones(scheduling_matrix[m].shape, device=self.device) * (len(time_steps) - 1)
            else:
                next_noise = scheduling_matrix[m+1]
                
            current_noise = time_steps[current_noise.int()]
            next_noise = time_steps[next_noise.int()]
            
            # add little noise to the context frames
            if with_context:
                context_current_noise = torch.ones(context_size, device=self.device) - choice(sigma)
                context_next_noise = torch.ones(context_size, device=self.device) - choice(sigma)
                current_noise = torch.cat((context_current_noise, current_noise), dim=0).to(window_x.device)
                next_noise = torch.cat((context_next_noise, next_noise), dim=0).to(window_x.device)
                
            # add batch dimension
            current_noise = repeat(current_noise, "t -> b t", b=window_x.shape[0])
            next_noise = repeat(next_noise, "t -> b t", b=window_x.shape[0])
                        
            window_x = self.flow_step(x_t=window_x, t_start=current_noise, t_end=next_noise, ctrl=window_ctrl)
        return window_x
    
    def on_validation_epoch_end(self):
        self.sample_imgs(self.val_outs, save=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def main(config: TrainingConfig):
    # Load model config
    with open(config.config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Setup logging
    run_name = model_config.get('RUN_NAME', 'flow_matching_run')
    wandb_logger = None
    if config.use_logger and not config.fast_check:
        wandb_logger = WandbLogger(
            project="diffusion_pain_flow_matching",
            name=run_name
        )
        wandb_logger.log_hyperparams(model_config)

    # Create necessary directories
    dirs = [
        model_config.get('DIFFUSION', {}).get('sample_output_dir', 'samples'),
        model_config.get('CHECKPOINT', 'checkpoints'),
        model_config.get('CODEBACKUP', 'codebackup')
    ]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    # Initialize data module
    biovid = BioVidDM.from_conf(config.config_path)
    biovid.setup('fit')

    # Initialize models
    net = VideoLatentUnet(**model_config['MODEL'])
    ctrl_emb = ControlEmbedding(
        emb_dim=128,
        in_dim=1,
        conf=model_config['DATA_STATS']
    )
    
    # Create flow matching module
    fm_model = FlowMatchingModule(
        net=net,
        ctrl_emb=ctrl_emb,
        lr=config.learning_rate,
        frame_stack=config.frame_stack,
        drop_probs=config.drop_probs,
        sample_output_dir=config.sample_output_dir,
        guide=model_config['DIFFUSION'].get('guide', [1, 1, 1])
    )

    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=model_config.get('CHECKPOINT', 'checkpoints'),
        filename='flow_matching-{epoch:02d}-{train_loss:.2f}',
        save_top_k=2,
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # EMA callback
    ema = EMA(
        decay=0.9999,
        evaluate_ema_weights_instead=True,
        save_ema_weights_in_callback_state=True
    )
    callbacks.append(ema)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    torch.set_float32_matmul_precision("highest")

    # Initialize trainer
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu',
        devices=config.devices if config.train else 1,
        strategy='ddp_find_unused_parameters_true' if config.devices > 1 else 'auto',
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        fast_dev_run=5 if config.fast_check else False,
    )

    # Training/validation/testing flow
    if config.train:
        trainer.fit(
            fm_model, 
            datamodule=biovid, 
            ckpt_path=config.checkpoint_path if config.load_from_checkpoint else None
        )

        # Save best checkpoint info
        best_ckpt = checkpoint_callback.best_model_path
        last_ckpt = checkpoint_callback.last_model_path

        if config.use_logger:
            wandb_logger.log_hyperparams({
                "best_ckpt": best_ckpt,
                'last_ckpt': last_ckpt
            })
            
        # Update config file with checkpoint paths
        with open(config.config_path, 'r') as f:
            conf = yaml.safe_load(f)
        
        conf['BEST_CKPT'] = best_ckpt
        conf['LAST_CKPT'] = last_ckpt
        
        with open(config.config_path, 'w') as f:
            yaml.safe_dump(conf, f)

    if config.validate:
        trainer.validate(
            fm_model, 
            datamodule=biovid,
            ckpt_path=config.checkpoint_path if config.load_from_checkpoint else None
        )

if __name__ == "__main__":
    tyro.cli(main)