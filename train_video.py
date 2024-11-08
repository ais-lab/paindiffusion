from lightning import Trainer
import torch
import yaml
from diffusion.module.utils.biovid import BioVidDM
from diffusion.elucidated_for_video import ElucidatedDiffusion

from lightning.pytorch.loggers import WandbLogger
import os

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from diffusion.module.utils.ema import EMA

if __name__ == "__main__":
        
    import argparse 
    
    parser = argparse.ArgumentParser(description='Main of training script')
    parser.add_argument('--conf', type=str, default="configure/video_conf.yml", help='Path to the configuration file')
    parser.add_argument('--load_from_checkpoint', action='store_true', help='Load from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--fast_check', action='store_true', help='Fast check')
    parser.add_argument('--logger', action='store_true', help='Use logger')
    
    args = parser.parse_args()
    
    conf_file = args.conf
    load_from_checkpoint = args.load_from_checkpoint
    checkpoint = args.checkpoint
    fast_check = args.fast_check
    logger = args.logger
    
    if fast_check:
        logger = False
        
        
    if load_from_checkpoint:
        assert checkpoint is not None, "Please provide the checkpoint path"
        
        
    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)
        
        run_name = conf['RUN_NAME']
        
        train = conf['TRAIN']
        validate = conf['VALIDATE']
        test = conf['TEST']
        
        wandb_logger = WandbLogger(project="diffusion_pain_emoca_latent_video", name=run_name) if logger else None

        if logger:
            wandb_logger.log_hyperparams(conf)
        
        dirs = [conf['DIFFUSION']['sample_output_dir'],conf['CHECKPOINT'], conf['CODEBACKUP']]
        
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)


    torch.set_float32_matmul_precision("highest") # use float32 matmul for better performance

    model = ElucidatedDiffusion.from_conf(conf_file)        

    biovid = BioVidDM.from_conf(conf_file)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=conf['CHECKPOINT'],
        filename='elucidated_diffusion-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        mode='min',
        save_last=True
    )


    ema = EMA(
        decay=0.999,
        evaluate_ema_weights_instead=True,
        save_ema_weights_in_callback_state=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Lightning Trainer for flexible accelerated training
    trainer = Trainer(
        max_epochs = 60,
        accelerator = 'gpu',
        devices = 2 if train else 1, # Piece of cake multi-gpu support!
        strategy = 'ddp_find_unused_parameters_true',
        logger=wandb_logger if logger else None,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, 
                ema, 
                lr_monitor],
        check_val_every_n_epoch=1,
        fast_dev_run=500 if fast_check else False,
    )

    if train:
        trainer.fit(model, datamodule = biovid, ckpt_path=checkpoint if load_from_checkpoint else None)

        best_ckpt = checkpoint_callback.best_model_path
        last_ckpt = checkpoint_callback.last_model_path

        if logger:
            wandb_logger.log_hyperparams({"best_ckpt": best_ckpt,
                                        'last_ckpt': last_ckpt})
            
        with open(conf_file, 'r') as f:
            conf = yaml.safe_load(f)
            
        with open(conf_file, 'w') as f:
            
            conf['BEST_CKPT'] = best_ckpt
            conf['LAST_CKPT'] = last_ckpt
            
            yaml.safe_dump(conf, f)

    if validate:
        trainer.validate(model, datamodule= biovid, ckpt_path=checkpoint if load_from_checkpoint else None)
        
    # if test:
    #     # each predict 200 videos
        
    #     # 128
    #     model.sample_output_dir = os.path.join(conf['DIFFUSION']['sample_output_dir'], "128")
    #     os.makedirs(model.sample_output_dir, exist_ok=True)
    #     trainer.test(model, datamodule= biovid, ckpt_path=checkpoint if load_from_checkpoint else None)
    #     # 640
    #     model.sample_output_dir = os.path.join(conf['DIFFUSION']['sample_output_dir'], "640")
    #     os.makedirs(model.sample_output_dir, exist_ok=True)
    #     trainer.test(model, datamodule= biovid, ckpt_path=checkpoint if load_from_checkpoint else None)
