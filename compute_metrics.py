import os
import torch
import numpy as np
from tqdm import tqdm
from run_baseline import get_val
from metrics.metrics import calculate_pain_metrics

def calculate_metrics(prediction_dir, max_try=5, max_sample=-1):
    
    # get valset
    val_set = get_val()
    
    # get vallist
    val_list = torch.load("val_list_bug.pt")
    gt_pspi_path = "/media/tien/SSD-NOT-OS/pain_intermediate_data/groundtruth/pspi"
    
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
        
    torch.save(new_val_list, f"val_list_with_old_idx.pt")
        
    val_list = new_val_list
    
    # predition list
    pred = {
        'exp': [],
        'pspi': [],
    }
    
    gt = {
        'exp': [],
        'pspi': [],
    }
    
    stimuli = []
    
    remove = True
    if remove and os.path.exists(f"{prediction_dir}/pred.pt"):
        os.remove(f"{prediction_dir}/pred.pt")
        os.remove(f"{prediction_dir}/gt.pt")
        os.remove(f"{prediction_dir}/stimuli.pt")
        

    
    if not os.path.exists(f"{prediction_dir}/pred.pt"):
        for try_idx in range(max_try):
            cnt = 0
            for idx, sample in tqdm(val_list):
                video_name, start_frame, end_frame = sample
                
                end_frame = start_frame + 608
                
                if not os.path.exists(f"{prediction_dir}/try_{try_idx}/{idx}.pt") and not os.path.exists(f"{prediction_dir}/try_{try_idx}/test_ctrl_{idx}.pt"):
                    continue
                
                try:
                    _pred = torch.load(f"{prediction_dir}/try_{try_idx}/{idx}.pt", map_location="cpu")
                except:
                    _pred = torch.load(f"{prediction_dir}/try_{try_idx}/test_ctrl_{idx}.pt", map_location="cpu")
                    
                cnt += 1
                
                try:
                    _pred = _pred['x']
                except:
                    _pred = _pred
                
                if len(_pred.shape) == 2: 
                    _pred = _pred.unsqueeze(0)
                    
                sequence_length = min(_pred.shape[1] if len(_pred.shape) == 3 else _pred.shape[0], 608)

                _pred = _pred[:,:sequence_length]
                pred['exp'].append(_pred)
                
                if not os.path.exists(f"{prediction_dir}/pspi_try_{try_idx}"):
                    continue
                
                try:
                    _pspi = torch.load(f"{prediction_dir}/pspi_try_{try_idx}/{idx}.pt")
                except:
                    _pspi = torch.load(f"{prediction_dir}/pspi_try_{try_idx}/test_ctrl_{idx}.pt")
                
                _pspi = [p[1] for p in _pspi]
                
                _pspi = _pspi[:sequence_length]
                
                pred['pspi'].append(_pspi)
                                    
                if try_idx == 0:
                    sample = val_set.__getitem__(idx, video_name=video_name, start_frame_id=start_frame, end_frame_id=end_frame)

                    exp_groundtruth = sample['x']
                    
                    exp_groundtruth[..., :3] /= 100
                    
                    _pspi_groundtruth = torch.load(os.path.join(gt_pspi_path, f"test_ctrl_{idx}.pt"))
                
                    pspi_groundtruth = [p[1] for p in _pspi_groundtruth]
                    
                    pspi_groundtruth = pspi_groundtruth[:sequence_length]
                    exp_groundtruth = exp_groundtruth[:sequence_length]
                
                    gt['exp'].append(exp_groundtruth)
                    gt['pspi'].append(torch.tensor(pspi_groundtruth))
                
                    _stimuli = sample['ctrl'][-2]
                    
                    _stimuli = _stimuli[:sequence_length]
                    
                    stimuli.append(_stimuli)
                
                if max_sample != -1 and idx == max_sample:
                    break
                    
    # print("metrics")
    
    if not os.path.exists(f"{prediction_dir}/pred.pt"):
        torch.save(pred, f"{prediction_dir}/pred.pt")
        torch.save(gt, f"{prediction_dir}/gt.pt")
        torch.save(stimuli, f"{prediction_dir}/stimuli.pt")
    else:
        pred = torch.load(f"{prediction_dir}/pred.pt")
        gt = torch.load(f"{prediction_dir}/gt.pt")
        stimuli = torch.load(f"{prediction_dir}/stimuli.pt")
    
    one_try_lenght = len(pred['exp']) // max_try
    
    print(f"one_try_lenght: {one_try_lenght}")
    multiple_exp = [torch.stack(pred['exp'][i:i+one_try_lenght]) for i in range(0, len(pred['exp']), one_try_lenght)]
    
    multiple_exp = torch.stack(multiple_exp).squeeze()
    # shape: (T, N, Seq-leng, D)

    # Handle PSPI predictions - may only have single try
    if len(pred['pspi']) > one_try_lenght:  # Multiple tries exist
        multiple_pspi = [torch.tensor(pred['pspi'][i:i+one_try_lenght]) for i in range(0, len(pred['pspi']), one_try_lenght)]
        multiple_pspi = torch.stack(multiple_pspi).squeeze()
    else:  # Single try
        multiple_pspi = torch.tensor(pred['pspi']).unsqueeze(0)
    # shape: (T, N, Seq-leng) or (1, N, Seq-leng) for single try
    
    exp_gt_tensor = torch.stack(gt['exp']).cpu()  # shape: (N, Seq-leng, D)
    pspi_gt_tensor = torch.stack(gt['pspi']).cpu() # shape: (N, Seq-leng)
    stimuli_tensor = torch.stack(stimuli).cpu()    # shape: (N, Seq-leng)
        
    exp_gt_tensor = exp_gt_tensor[..., :103]
    ensemble_exp = multiple_exp.cpu()[..., :103]
    
    metrics_per_try = []
    
    def compute_metrics_for_try(i, multiple_exp, ensemble_exp, exp_gt_tensor, multiple_pspi, pspi_gt_tensor, stimuli_tensor):
        exp_pred_i = multiple_exp[i].cpu()[..., :103]
        pspi_pred_i = multiple_pspi[min(i, len(multiple_pspi)-1)].cpu()  # Use min to handle single try case
        return calculate_pain_metrics(
            exp_pred_i,
            ensemble_exp,
            exp_gt_tensor,
            pspi_pred_i,
            pspi_gt_tensor,
            stimuli_tensor
        )

    compute_fn = partial(
        compute_metrics_for_try,
        multiple_exp=multiple_exp,
        ensemble_exp=ensemble_exp,
        exp_gt_tensor=exp_gt_tensor,
        multiple_pspi=multiple_pspi,
        pspi_gt_tensor=pspi_gt_tensor,
        stimuli_tensor=stimuli_tensor
    )

    with ThreadPoolExecutor(max_workers=20) as executor:
        metrics_per_try = list(executor.map(compute_fn, range(max_try)))
        
    metrics_array = np.array(metrics_per_try)
    
    metric_names = ["pain_dist", "pain_divrs", "pain_var", "pain_corr", "pain_sim", "pain_acc"]
    for i, name in enumerate(metric_names):
        m = metrics_array[:, i].mean()
        v = metrics_array[:, i].var()
        print(f"Metric {name}: mean = {m:.4f}, variance = {v:.4f}")
        with open(f"{prediction_dir}/metrics.txt", "a") as f:
            f.write(f"Metric {name}: mean = {m:.4f}, variance = {v:.4f}\n")
            
                
    
    
    
if __name__ == "__main__":
    
    import argparse 
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from concurrent.futures import ThreadPoolExecutor
    
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/eval_output")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/baseline/3dmm")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/without_diffusion_forcing")
    
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/context_window/2")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/context_window/4")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/context_window/8")
    
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/df_uncertainty/0_5")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/df_uncertainty/1")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/df_uncertainty/2")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/df_uncertainty/4")
    
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/guiding/1_1_1")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/guiding/1_2_4")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/guiding/05_1_2")
    # parser.add_argument("--prediction_dir", type=str, default="/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/guiding/025_05_1")
    
    
    dirs = [
        
    #     "/media/tien/SSD-NOT-OS/baseline_new/with_old_ind/",
    # "/media/tien/SSD-NOT-OS/pain_intermediate_data/eval_output",
    # "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/without_diffusion_forcing",
    # "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/context_window/2",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/context_window/4",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/context_window/8",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/df_uncertainty/0_5",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/df_uncertainty/1",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/df_uncertainty/2",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/df_uncertainty/4",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/guiding/1_1_1",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/guiding/1_2_4",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/guiding/05_1_2",
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/guiding/025_05_1",
    
    # "/media/tien/SSD-NOT-OS/pain_intermediate_data/baseline_new/with_old_ind",
    # "/media/tien/SSD-NOT-OS/pain_intermediate_data/eval_output_new",
    # "/media/tien/SSD-NOT-OS/pain_intermediate_data/ablation/without_diffusion_forcing",
    
    ]
    
    parser.add_argument("--max_try", type=int, default=5)
    
    # parser.add_argument("--max_sample", type=int, default=-1)
    parser.add_argument("--max_sample", type=int, default=100)
    
    
    args = parser.parse_args()
    
    # prediction_dir = args.prediction_dir
    
    for dir in dirs:
        print(f"Processing directory: {dir}")
        # Your processing logic here
    
        calculate_metrics(dir, max_try=args.max_try, max_sample=args.max_sample)