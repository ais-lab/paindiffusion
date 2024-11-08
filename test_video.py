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

import yaml


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Main of test script")
    parser.add_argument(
        "--conf",
        type=str,
        default="/home/tien/fr2-pain/configure/ablation_framestack_4.yml",
        help="Path to the configuration file",
    )
    parser.add_argument("--fast_check", action="store_true", help="Fast check")

    args = parser.parse_args()

    conf_file = args.conf
    fast_check = args.fast_check

    with open(conf_file, "r") as f:
        conf = yaml.safe_load(f)

        run_name = conf["RUN_NAME"]

        train = conf["TRAIN"]
        validate = conf["VALIDATE"]
        test = conf["TEST"]


        dirs = [
            conf["DIFFUSION"]["sample_output_dir"],
            conf["CHECKPOINT"],
            conf["CODEBACKUP"],
        ]

        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

        best_checkpoint = conf["BEST_CKPT"]

    torch.set_float32_matmul_precision("highest")

    model = ElucidatedDiffusion.from_conf(conf_file)

    # Lightning Trainer for flexible accelerated training
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        strategy="ddp_find_unused_parameters_true",
        fast_dev_run=500 if fast_check else False,
    )

    model.sample_output_dir = os.path.join(
        conf["DIFFUSION"]["sample_output_dir"], "128"
    )
    os.makedirs(model.sample_output_dir, exist_ok=True)
    
    with open(os.path.join(model.sample_output_dir, "config.yml"), "w") as f:
        conf['DATASET']['test_max_length'] = 128
        yaml.dump(conf, f)
        
    biovid = BioVidDM.from_conf(conf_file)
    
    trainer.test(model, datamodule=biovid, ckpt_path=best_checkpoint)
    
    model.sample_output_dir = os.path.join(
        conf["DIFFUSION"]["sample_output_dir"], "640"
    )
    
    with open(os.path.join(model.sample_output_dir, "config.yml"), "w") as f:
        conf['DATASET']['test_max_length'] = 640
        yaml.dump(conf, f)
        
    biovid = BioVidDM.from_conf(conf_file)
    
    os.makedirs(model.sample_output_dir, exist_ok=True)
    trainer.test(model, datamodule=biovid, ckpt_path=best_checkpoint)
