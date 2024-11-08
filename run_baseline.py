import os
import random
from baselines.mean_static_face import mean_static_face
from baselines.nearest_neighbor import nearest_neighbor
from baselines.random_sequence_dataset import random_sequence
from diffusion.module.utils.biovid import BioVidDataset

# from preprocess.extract_pspi import get_pspi_from_video
# from einops import rearrange
import torch

# from tqdm import tqdm

from metrics.metrics import calculate_pain_metrics

from metrics.metrics import Metrics

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
    max_length=608,
    is_video=True,
    )
    
    val_dataset.temp_dir = temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    # for sample in val_dataset:
    #     yield sample
    
    return val_dataset
    
        
if __name__ == "__main__":
    
    from tqdm import tqdm
    
    # set seed for everything
    
    output_path = "/media/tien/SSD-NOT-OS/pain_intermediate_data/output_baseline/"
    
    gt_pspi_path = "/media/tien/SSD-NOT-OS/pain_intermediate_data/groundtruth/pspi/"
    
    val_list = torch.load("val_list_bug.pt")
    
    correct_val_list_ = torch.load("val_list.pt")
    correct_val_list = [(video_name, start_frame.cpu().item(), end_frame.cpu().item()) for video_name, start_frame, end_frame in correct_val_list_]
    correct_val_list = set(correct_val_list)
    
    new_val_list = []
    for idx, sample in enumerate(tqdm(val_list)):
        
        video_name, start_frame, end_frame = sample
        
        if (video_name, start_frame.cpu().item(), end_frame.cpu().item()) not in correct_val_list:
            # print("skip", video_name, start_frame, end_frame)
            continue
        
        new_val_list.append((idx, sample))
        
    print("new_val_list", len(new_val_list))
    
    val_set = get_val()
    
    print("predicting exp")
    
    # mean_baseline = {}
    nn_baseline = {
        'exp': [],
        'pspi': [],
    }
    random_baseline = {
        'exp': [],
        'pspi': [],
    }
    
    gt = {
        'exp': [],
        'pspi': [],
    }
    
    stimuli = []
    
    final_idx = 0
    for try_idx in range(1):
        for idx, sample in tqdm(new_val_list):
            
            video_name, start_frame, end_frame = sample
            
            end_frame = start_frame + 608
                        
        #     # save prediction for each baseline
            
            # mean_static_face
            # exp_mean_prediction = mean_static_face(sample['x'].shape[0])
            # pspi_mean_prediction = ... # TODO
            
            sample = val_set.__getitem__(idx, video_name=video_name, start_frame_id=start_frame, end_frame_id=end_frame)
            
            # nearest_neighbor
            nn_prediction = nearest_neighbor(sample)
            exp_nn_prediction = nn_prediction['x']
            exp_nn_prediction[..., :3] /= 100
            pspi_nn_predition = nn_prediction['ctrl'][-1]
            
            nn_baseline['exp'].append(exp_nn_prediction)                
            
        #     # random_sequence
            random_prediction = random_sequence()
            exp_random_prediction = random_prediction['x']
            exp_random_prediction[..., :3] /= 100
            pspi_random_prediction = random_prediction['ctrl'][-1]
            
            random_baseline['exp'].append(exp_random_prediction)
            
            if try_idx == 0:
                
                nn_baseline['pspi'].append(pspi_nn_predition)

                random_baseline['pspi'].append(pspi_random_prediction)
            
                exp_groundtruth = sample['x']
                
                _pspi_groundtruth = torch.load(os.path.join(gt_pspi_path, f"test_ctrl_{idx}.pt"))
                
                pspi_groundtruth = [p[1] for p in _pspi_groundtruth]
                
                # pspi_groundtruth = sample['ctrl'][-1]
            
                gt['exp'].append(exp_groundtruth)
                gt['pspi'].append(torch.tensor(pspi_groundtruth))
            
                _stimuli = sample['ctrl'][-2]
            
                stimuli.append(_stimuli)
                            
    # calculate_pain_metrics(exp_pred, exp_multiple, exp_gt, pspi_pred, pspi_gt, stimuli)
    
    # backup the object
    # torch.save(nn_baseline, os.path.join(output_path, "nn_baseline.pt"))
    # torch.save(random_baseline, os.path.join(output_path, "random_baseline.pt"))
    # torch.save(gt, os.path.join(output_path, "gt.pt"))
    # torch.save(stimuli, os.path.join(output_path, "stimuli.pt"))
    
    # nn_baseline = torch.load(os.path.join(output_path, "nn_baseline.pt"))
    # random_baseline = torch.load(os.path.join(output_path, "random_baseline.pt"))
    # gt = torch.load(os.path.join(output_path, "gt.pt"))
    # stimuli = torch.load(os.path.join(output_path, "stimuli.pt"))
    
    from einops import rearrange
    
    one_try_lenght = len(new_val_list)
    nn_multiple_exp = [torch.stack(nn_baseline['exp'][i:i+one_try_lenght]) for i in range(0, len(nn_baseline['exp']), one_try_lenght)]
    nn_multiple_exp = torch.stack(nn_multiple_exp)
    # print(nn_multiple_exp.shape)
    # print(nn_multiple_exp[0][0][0][:10])
    # print(nn_multiple_exp[1][0][0][:10])
    
    random_multiple_exp = [torch.stack(random_baseline['exp'][i:i+one_try_lenght]) for i in range(0, len(random_baseline['exp']), one_try_lenght)]
    random_multiple_exp = torch.stack(random_multiple_exp)
    # print(random_multiple_exp.shape)
    # print(random_multiple_exp[0][0][0][:10])
    # print(random_multiple_exp[1][0][0][:10])
    
    # print("calculating nn metrics")
    # nn_metrics = calculate_pain_metrics(
    #     torch.stack(nn_baseline['exp'][:len(new_val_list)]),
    #     nn_multiple_exp,
    #     torch.stack(gt['exp']),
    #     torch.stack(nn_baseline['pspi']),
    #     torch.stack(gt['pspi'])[...,:608],
    #     torch.stack(stimuli),
    # )
    
    print("calculating random metrics")
    random_metrics = calculate_pain_metrics(
        torch.stack(random_baseline['exp'][:len(new_val_list)]),
        random_multiple_exp,
        torch.stack(gt['exp']),
        torch.stack(random_baseline['pspi']),
        torch.stack(gt['pspi'])[...,:608],
        torch.stack(stimuli),
    )
    