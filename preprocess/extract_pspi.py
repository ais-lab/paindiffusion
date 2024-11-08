import os
from preprocess.MEGraphAU import MEFARG
from preprocess.MEGraphAU.utils import load_state_dict
import cv2
import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
    ]
)

au_bp4d = MEFARG(num_classes=12, backbone="swin_transformer_base")
au_path = "/home/tien/fr2-pain/preprocess/MEGraphAU/checkpoints/ME-GraphAU_swin_base_BP4D/MEFARG_swin_base_BP4D_fold1.pth"
au_bp4d_net = load_state_dict(au_bp4d, au_path)
au_bp4d_net = au_bp4d_net.to(device=device)
au_bp4d_net.eval()

au_disfa = MEFARG(num_classes=8, backbone="swin_transformer_base")
au_path = "/home/tien/fr2-pain/preprocess/MEGraphAU/checkpoints/ME-GraphAU_swin_base_DISFA/MEFARG_swin_base_DISFA_fold1.pth"
au_disfa_net = load_state_dict(au_disfa, au_path)
au_disfa_net = au_disfa_net.to(device=device)
au_disfa_net.eval()


# batch 

def get_pspi(batch):

    # batch = torch.randn(1, 224, 224, 3).to(device=device)
    b, *_ = batch.shape

    au_score, _ = au_bp4d_net(batch)
    au_score_disfa, _ = au_disfa_net(batch)
    
    pspi = []
    
    for idx in range(b):

        _au_score = au_score[idx].tolist()

        _au_disfa_score = au_score_disfa[idx].tolist()

        au4 = _au_score[2]
        au6 = _au_score[3]
        au7 = _au_score[4]
        au10 = _au_score[5]

        au9 = _au_disfa_score[4]

        # rescale au to 0-5 range
        au4 = au4 * 5
        au6 = au6 * 5
        au7 = au7 * 5
        au9 = au9 * 5
        au10 = au10 * 5

        pspi_no_au43 = au4 + max(au6, au7) + max(au9, au10)
        
        pspi.append(pspi_no_au43)
        
    return pspi


def get_pspi_from_video(path_to_video):
    
    frame_path = os.listdir(path_to_video)
    batch_size = 12
    batch = []
    batch_name = []
    pspi = []
    for idx, frame in enumerate(frame_path):
        frame = os.path.join(path_to_video, frame)
        # batch = torch.randn(1, 224, 224, 3).to(device=device)
        
        image = cv2.imread(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_transform(image) 
        
        batch_name.append(frame)
        batch.append(image)
        
        if len(batch) == batch_size or idx == len(frame_path) - 1:
            
            batch = torch.stack(batch)
            
            batch = batch.to(device=device)
        
            _pspi = get_pspi(batch)
            
            pspi.extend(zip(batch_name, _pspi))
            
            batch = []
            batch_name = []
            
    pspi = sorted(pspi, key=lambda x: int(x[0].split("/")[-1].split(".")[0]))
            
    return pspi


if __name__ == "__main__":
    
    video_dir = "/home/tien/inferno/out1"
    
    pspi = get_pspi_from_video(video_dir)
    
    # ('/home/tien/inferno/out2/7.jpg', 9.410572350025177)
    # sort pspi by name of frame 
        
    assert len(pspi) == len(os.listdir(video_dir))
    
    # save pspi to file in the same dir as video_dir
    with open(os.path.join(video_dir, "pspi.txt"), "w") as f:
        for frame, score in pspi:
            f.write(f"{frame} {score}\n")
            