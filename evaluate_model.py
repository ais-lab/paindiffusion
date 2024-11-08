import os
import torch
from tqdm import tqdm

from diffusion.module.utils.biovid import BioVidDataset


def get_val():
    
    path_to_frame_labels = "/media/tien/SSD-NOT-OS/pain_intermediate_data/processed_pain_data_no_facedetector/"
    path_to_video_frame = (
        "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/extracted_frame/"
    )
    path_to_3d_latents = (
        "/media/tien/SSD-NOT-OS/pain_intermediate_data/emoca_latent_code/"
    )
    temp_dir = "/media/tien/SSD-NOT-OS/pain_intermediate_data/temp_video_eval"
    
    val_dataset = BioVidDataset(
    path_to_video_frame=path_to_video_frame,
    path_to_frame_labels=path_to_frame_labels,
    path_to_3d_latents=path_to_3d_latents,
    split="val",
    max_length=640,
    is_video=True,
    )
    
    val_dataset.temp_dir = temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    # for sample in val_dataset:
    #     yield sample
    
    return val_dataset


def add_batch(sample):
    for key in sample:
        if isinstance(sample[key], torch.Tensor):
            sample[key] = sample[key].unsqueeze(0).cuda()
        if isinstance(sample[key], list):
            sample[key] = [x.unsqueeze(0).cuda() for x in sample[key]]
    return sample

if __name__ == "__main__":
    
    from lightning import Trainer
    import yaml
    from diffusion.elucidated_for_video import ElucidatedDiffusion
    from diffusion.module.utils.biovid import BioVidDM

    conf_file = "/home/tien/fr2-pain/configure/ablation_context_2.yml"
    output_path = "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/context_window/2"
    max_try = 5
    max_sample = -1
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", type=str, default=conf_file)
    parser.add_argument("--output_path", type=str, default=output_path)
    parser.add_argument("--max_try", type=int, default=max_try)
    parser.add_argument("--max_sample", type=int, default=max_sample)
    
    # example
    # python evaluate_model.py --conf_file /home/tien/fr2-pain/configure/ablation_context_2.yml --output_path /media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/context_window/2 --max_try 5 --max_sample -1   
     
    args = parser.parse_args()
    
    conf_file = args.conf_file
    output_path = args.output_path
    max_try = args.max_try
    max_sample = args.max_sample

    model = ElucidatedDiffusion.from_conf(conf_file)

    with open(conf_file, "r") as f:
        conf = yaml.safe_load(f)

    best_checkpoint = conf["BEST_CKPT"]

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        fast_dev_run=1,
        logger=False,
    )

    biovid = BioVidDM.from_conf(conf_file)

    # model = ElucidatedDiffusion.load_from_checkpoint(best_checkpoint, model=model, conf=conf)

    trainer.test(model, datamodule=biovid, ckpt_path=best_checkpoint)

    model.eval()

    model = model.to("cuda")
    
    os.makedirs(output_path, exist_ok=True)
    
    val_list = torch.load("val_list.pt")
    
    val_set = get_val()
    
    # model_pred = {
    #     'exp': [],
    #     'pspi': [],
    # }
    
    batch_size = 10
        
    def add_to_batch(sample):
        for key in batch:
            batch[key].append(sample[key])
            
    def add_batch(sample):
        for key in sample:
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].unsqueeze(0).cuda()
            if isinstance(sample[key], list):
                sample[key] = [x.unsqueeze(0).cuda() for x in sample[key]]
        return sample
            
    def stack_the_batch(batch):
        for key in batch:
            if isinstance(batch[key][0], torch.Tensor): # x
                batch[key] = torch.stack(batch[key])
                batch[key] = batch[key].to("cuda")
            if isinstance(batch[key][0], list): # ctrl
                ctrl = []
                for feature_id in range(len(batch[key][0])):
                    feature = torch.stack([x[feature_id] for x in batch[key]])
                    feature = feature.to("cuda")
                    ctrl.append(feature)
                batch[key] = ctrl
    
    for try_id in range(max_try):
        
        print(f"try_id: {try_id}")
        
        batch = {
            'x': [],
            'ctrl': [],
        }
        checkpoint = 0
        
        os.makedirs(os.path.join(output_path, f"try_{try_id}"), exist_ok=True)

        for idx, sample in tqdm(enumerate(val_list)):
            
            video_name, start_frame, end_frame = sample
            
            end_frame = start_frame + 640
            
            sample = val_set.__getitem__(idx, video_name=video_name, start_frame_id=start_frame, end_frame_id=end_frame)
            
            # add_to_batch(sample)
            
            sample = add_batch(sample)  
            
            # if len(batch['x']) == batch_size:
                                
                # stack_the_batch(batch)
                
            frame_prediction = model.sample_imgs(batch=sample, save=False)
            
            torch.save(frame_prediction['x'], os.path.join(output_path, f"try_{try_id}", f"{idx}.pt"))
            
            if max_sample != -1 and idx >= max_sample:
                break
                
            # for pred_id, _ in enumerate(frame_prediction['x']):
            # real_id = checkpoint + pred_id
            # os.makedirs(os.path.join(output_path, f"try_{try_id}"), exist_ok=True)
            # torch.save(frame_prediction['x'][pred_id], os.path.join(output_path, f"try_{try_id}", f"{real_id}.pt"))    
                    
                # batch = {
                #     'x': [],
                #     'ctrl': [],
                # }
                
                # checkpoint = idx + 1