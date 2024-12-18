import json
import os
import random
import shutil

# import pandas as pd
import numpy as np
import scipy
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms

from .data import AbstractDM

def exists(var):
    return var is not None

def default(var, val ):
    return var if exists(var) else val

temp_dir = "/media/tien/SSD-NOT-OS/pain_intermediate_data/temp_video"
# shutil.rmtree(temp_dir, ignore_errors=True)

def savitzky_golay(original_data, window_length=10, polyorder=2):
    """
    Savitzky-Golay filter for smoothing
    Preserves higher moments of the signal
    
    Parameters:
    - window_length: Length of the filter window
    - polyorder: Order of the polynomial used for fitting
    
    Returns:
    - Smoothed data with same shape as input
    """
    smoothed = np.zeros_like(original_data)
    for i in range(original_data.shape[1]):
        smoothed[:, i] = scipy.signal.savgol_filter(
            original_data[:, i], 
            window_length=window_length, 
            polyorder=polyorder
        )
    return smoothed

class BioVidDataset(Dataset):

    idx2emotion = {
        0: "Anger",
        1: "Contempt",
        2: "Disgust",
        3: "Fear",
        4: "Happiness",
        5: "Neutral",
        6: "Sadness",
        7: "Surprise",
    }

    emotion2idx = {
        "Anger": 0,
        "Contempt": 1,
        "Disgust": 2,
        "Fear": 3,
        "Happiness": 4,
        "Neutral": 5,
        "Sadness": 6,
        "Surprise": 7,
    }

    # follow split by https://www.nit.ovgu.de/nit_media/Bilder/Dokumente/BIOVID_Dokumente/BioVid_HoldOutEval_Proposal.pdf
    validation_idetities = [
        "100914_m_39",
        "101114_w_37",
        "082315_w_60",
        "083114_w_55",
        "083109_m_60",
        "072514_m_27",
        "080309_m_29",
        "112016_m_25",
        "112310_m_20",
        "092813_w_24",
        "112809_w_23",
        "112909_w_20",
        "071313_m_41",
        "101309_m_48",
        "101609_m_36",
        "091809_w_43",
        "102214_w_36",
        "102316_w_50",
        "112009_w_43",
        "101814_m_58",
        "101908_m_61",
        "102309_m_61",
        "112209_m_51",
        "112610_w_60",
        "112914_w_51",
        "120514_w_56",
    ]

    def __init__(
        self,
        path_to_frame_labels: str,
        path_to_video_frame: str,
        path_to_3d_latents: str,
        
        #TODO: parameterize this
        path_to_change_states: str = "preprocess/change_state.json",
        
        max_length: int = 256,
        img_size: int = 256,
        load_au_features: bool = True,
        load_emotion_labels: bool = True,
        load_stimulus_values: bool = True,
        load_stimulus_label: bool = True,
        load_pspi_no_au43: bool = True,
        load_3d_latents: bool = True,
        load_frame: bool = False,
        split: str = "train",
        is_video: bool = False,
        smooth_latent: bool = False,
    ):

        self.path_to_3d_latents = path_to_3d_latents
        self.video_frame_dir = path_to_video_frame
        self.frame_labels_dir = path_to_frame_labels

        self.video_names = os.listdir(self.video_frame_dir)

        self.load_au_features = load_au_features
        self.load_emotion_labels = load_emotion_labels
        self.load_stimulus_values = load_stimulus_values
        self.load_stimulus_label = load_stimulus_label
        self.load_pspi_no_au43 = load_pspi_no_au43
        self.load_3d_latents = load_3d_latents
        self.load_frame = load_frame
        
        self.smooth_latent = smooth_latent

        self.video_chunks = []
        self.max_length = max_length
        
        with open(path_to_change_states, "r") as f:
            self.change_states = json.load(f)
            
        self.temp_dir = temp_dir
            
        if os.path.exists(temp_dir):
            print("WARNING: temp dir exists, if any change was made to the dataset, please remove the temp dir first!!")
            
        os.makedirs(temp_dir, exist_ok=True)
        
        print("WARNING: the jawpose is scaled by 100, to render the output correctly, please scale it back by 100, \nCheck biovid.py:322")

        if load_frame:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            self._transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToDtype(torch.float32, scale=True),
                    normalize,
                ]
            )
            
        if split == "train":
            self.video_names = list(
                set(self.video_names) - set(BioVidDataset.validation_idetities)
            )
            
            self.video_names = sorted(self.video_names)
        elif split == "val":
            self.video_names = BioVidDataset.validation_idetities
            # DONE: only validate on sub sequence that have changing states

        for video_name in self.video_names:

            # random_start = random.randint(1, max_length)
            random_start = 1

            video_length = len(
                os.listdir(os.path.join(self.video_frame_dir, video_name))
            )
            
            changing_state_frames = self.change_states[video_name]
            
            if split == "train":

                # for chunk_start_pointer in range(
                #     random_start, 
                #     video_length + 1, 
                #     10 # self.max_length
                # ):
                #     if chunk_start_pointer + self.max_length > video_length:
                #         continue
                #     self.video_chunks.append(
                #         (
                #             video_name,
                #             chunk_start_pointer,
                #             min(chunk_start_pointer + self.max_length, video_length),
                #         )
                #     )
                
                for change_state_frame in changing_state_frames:
                    random_start = random.randint(0, 9)
                    
                    chunk_start_pointer = change_state_frame[0] - int(100*(1/4)) + random_start
                    chunk_end_pointer = change_state_frame[0] + int(200*(3/4)) + random_start
                    
                    if chunk_start_pointer < 1:
                        chunk_start_pointer = 1
                        
                    if chunk_end_pointer > video_length:
                        chunk_end_pointer = video_length
                    
                    for _chunk_start_pointer in range(
                        chunk_start_pointer,
                        chunk_end_pointer,
                        10
                    ):
                        if _chunk_start_pointer + self.max_length > video_length:
                            continue
                        self.video_chunks.append(
                            (
                                video_name,
                                _chunk_start_pointer,
                                min(_chunk_start_pointer + self.max_length, video_length),
                            )
                        )
                
            
            elif split == "val":
                
                # list of tuple (frame_id, prev_state, later_state)
                
                for change_state_frame in changing_state_frames:
                    chunk_start_pointer = change_state_frame[0] - int(self.max_length*(1/4))
                    chunk_end_pointer = change_state_frame[0] + int(self.max_length*(3/4))
                    
                    if chunk_start_pointer < 1:
                        chunk_start_pointer = 1
                        chunk_end_pointer = chunk_start_pointer + self.max_length
                        
                    if chunk_end_pointer > video_length:
                        chunk_end_pointer = video_length
                        chunk_start_pointer = chunk_end_pointer - self.max_length
                        
                    if chunk_end_pointer - chunk_start_pointer != self.max_length:
                        if chunk_end_pointer < video_length:
                            chunk_end_pointer = chunk_start_pointer + self.max_length
                        else:
                            chunk_start_pointer = chunk_end_pointer - self.max_length
                        
                        
                    self.video_chunks.append(
                        (
                            video_name,
                            chunk_start_pointer,
                            chunk_end_pointer,
                        )
                    )
                
        self._len = len(self.video_chunks)
        
        self.split = split
        
        self.is_video = is_video
        
        # self.video_chunks = pd.DataFrame(self.video_chunks, columns=['video_name', 'start_pointer', 'end_of_chunk'])

    def __len__(self):
        return self._len

    def __getitem__(self, idx, video_name=None, start_frame_id=None, end_frame_id=None):

        
        if video_name is not None:
            video_name = video_name
            start_frame_id = start_frame_id
            end_frame_id = end_frame_id
        
        else:
            video_name, start_frame_id, end_frame_id = self.video_chunks[idx]


        frames = []
        au_features = []
        emotion_labels = []
        stimulus_abs = []
        stimulus_cls = []
        pspi_no_au43 = []
        latent_3d = []
        
        # cache
        # if os.path.exists(os.path.join(self.temp_dir, f"{video_name}_{start_frame_id}_{end_frame_id}.pt")) and self.split == "train":
        #     # cache hit
        #     return torch.load(os.path.join(self.temp_dir, f"{video_name}_{start_frame_id}_{end_frame_id}.pt"), map_location="cpu")

        # cache miss
        for frame_id in range(start_frame_id, end_frame_id):
        # DONE: this loop is the botteckneck, need to optimize this

            if self.load_frame:
                frame_name = os.path.join(
                    self.video_frame_dir, video_name, f"frame_{frame_id}.jpg"
                )

                frame = read_image(frame_name)
                frame = self._transform(frame)
                frames.append(frame.unsqueeze(0))

            if (
                self.load_au_features
                or self.load_emotion_labels
                or self.load_stimulus_values
                or self.load_stimulus_label
                or self.load_pspi_no_au43
            ):
                frame_label_name = os.path.join(
                    self.frame_labels_dir, video_name, f"{frame_id}.json"
                )
                with open(frame_label_name, "r") as f:
                    frame_label = json.load(f)

                if self.load_au_features:
                    au_feature = frame_label["au_bp4d_score"]
                    au_feature.extend(frame_label["au_disfa_score"])
                    au_features.append(au_feature)

                if self.load_emotion_labels:
                    emotion_labels.append(
                        BioVidDataset.emotion2idx[frame_label["emotion"]]
                    )

                if self.load_stimulus_values:
                    stimulus_abs.append(frame_label["temperature"])

                if self.load_stimulus_label:
                    stimulus_cls.append(frame_label["label"])

                if self.load_pspi_no_au43:
                    pspi_no_au43.append(frame_label["pspi_no_au43"])

            if self.load_3d_latents:

                latent_name = os.path.join(
                    self.path_to_3d_latents, video_name, f"{frame_id}.json"
                )
                with open(latent_name, "r") as f:
                    latent_dict = json.load(f)
                
                # scale jawpose by 100
                latent_vector = torch.cat([torch.tensor(latent_dict["jawpose"])*100, torch.tensor(latent_dict["expcode"])], dim=0)

                latent_3d.append(latent_vector)

        if len(frames) > 0:
            frames = torch.cat(frames, dim=0)

        if len(au_features) > 0:
            au_features = torch.tensor(au_features)

        if len(emotion_labels) > 0:
            emotion_labels = torch.tensor(emotion_labels)

        if len(stimulus_abs) > 0:
            stimulus_abs = torch.tensor(stimulus_abs)

        if len(stimulus_cls) > 0:
            stimulus_cls = torch.tensor(stimulus_cls)

        if len(pspi_no_au43) > 0:
            pspi_no_au43 = torch.tensor(pspi_no_au43)
            
        if len(latent_3d) > 0:
            latent_3d = torch.stack(latent_3d)
            
            if self.smooth_latent:
                latent_3d = savitzky_golay(latent_3d)
                latent_3d = torch.tensor(latent_3d)
                    
        desire_dim = 128 # NOTE: parameterize this later
        
        current_latent_dim = latent_3d.shape[-1]
        
        # pad the latent 3d to a power of 2
        latent_3d = F.pad(latent_3d, (0, desire_dim-current_latent_dim), mode="constant", value=0)
        
        control_emotion = torch.mean(emotion_labels.float())
        pain_expressiveness = torch.mean(pspi_no_au43.float())
        
        # repeat the values to match the length of the latent
        control_emotion = control_emotion.repeat(latent_3d.size(0)).contiguous()
        pain_expressiveness = pain_expressiveness.repeat(latent_3d.size(0)).contiguous()
        
        if not self.is_video:
        
            current_latent = latent_3d[-1]
            past_latent = latent_3d[:-1]
            
            output = {
                "ctrl": [control_emotion, pain_expressiveness, stimulus_abs, past_latent],
                "x": current_latent.unsqueeze(0),
            }
            
        else:
            
            output = {
                "ctrl": [control_emotion, pain_expressiveness, stimulus_abs, pspi_no_au43],
                "x": latent_3d, # t, [c], d -> add channel dimension, 
                "video_name": video_name,
                "start_frame_id": start_frame_id,
                "end_frame_id": end_frame_id,
                "stimulus_cls": stimulus_cls,
            }
        
        # save the return to a temp file for optimizing
        temp_file = os.path.join(self.temp_dir, f"{video_name}_{start_frame_id}_{end_frame_id}.pt")
        # torch.save(output,
        #     temp_file,
        # )

        return output


class BioVidDM(AbstractDM):
    def __init__(
        self,
        path_to_frame_labels: str,
        path_to_video_frame: str,
        path_to_3d_latents: str,
        max_length: int = 256,
        test_max_length: int = 640,
        val_max_length: int = 640,
        img_size: int = 256,
        load_au_features: bool = True,
        load_emotion_labels: bool = True,
        load_stimulus_values: bool = True,
        load_stimulus_label: bool = True,
        load_pspi_no_au43: bool = True,
        load_3d_latents: bool = True,
        load_frame: bool = False,
        is_video: bool = False,
        smooth_latent: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.path_to_frame_labels = path_to_frame_labels
        self.path_to_video_frame = path_to_video_frame
        self.path_to_3d_latents = path_to_3d_latents
        self.max_length = max_length
        self.img_size = img_size
        self.load_au_features = load_au_features
        self.load_emotion_labels = load_emotion_labels
        self.load_stimulus_values = load_stimulus_values
        self.load_stimulus_label = load_stimulus_label
        self.load_pspi_no_au43 = load_pspi_no_au43
        self.load_3d_latents = load_3d_latents
        self.load_frame = load_frame
        self.is_video = is_video
        self.test_max_length = test_max_length
        self.smooth_latent = smooth_latent
        self.val_max_length = default(val_max_length, test_max_length)

    def setup(self, stage: str):
        biovid_train = BioVidDataset(
            path_to_frame_labels=self.path_to_frame_labels,
            path_to_video_frame=self.path_to_video_frame,
            path_to_3d_latents=self.path_to_3d_latents,
            max_length=self.max_length,
            img_size=self.img_size,
            load_au_features=self.load_au_features,
            load_emotion_labels=self.load_emotion_labels,
            load_stimulus_values=self.load_stimulus_values,
            load_stimulus_label=self.load_stimulus_label,
            load_pspi_no_au43=self.load_pspi_no_au43,
            load_3d_latents=self.load_3d_latents,
            load_frame=self.load_frame,
            split="train",
            is_video=self.is_video,
            smooth_latent=self.smooth_latent,
        )

        biovid_val = BioVidDataset(
            path_to_frame_labels=self.path_to_frame_labels,
            path_to_video_frame=self.path_to_video_frame,
            path_to_3d_latents=self.path_to_3d_latents,
            max_length=self.val_max_length,
            img_size=self.img_size,
            load_au_features=self.load_au_features,
            load_emotion_labels=self.load_emotion_labels,
            load_stimulus_values=self.load_stimulus_values,
            load_stimulus_label=self.load_stimulus_label,
            load_pspi_no_au43=self.load_pspi_no_au43,
            load_3d_latents=self.load_3d_latents,
            load_frame=self.load_frame,
            split="val",
            is_video=self.is_video,
            smooth_latent=self.smooth_latent,
        )
        
        biovid_test = BioVidDataset(
            path_to_frame_labels=self.path_to_frame_labels,
            path_to_video_frame=self.path_to_video_frame,
            path_to_3d_latents=self.path_to_3d_latents,
            max_length=self.test_max_length,
            img_size=self.img_size,
            load_au_features=self.load_au_features,
            load_emotion_labels=self.load_emotion_labels,
            load_stimulus_values=self.load_stimulus_values,
            load_stimulus_label=self.load_stimulus_label,
            load_pspi_no_au43=self.load_pspi_no_au43,
            load_3d_latents=self.load_3d_latents,
            load_frame=self.load_frame,
            split="val",
            is_video=self.is_video,
            smooth_latent=self.smooth_latent,
        )

        if stage == "fit" or stage=="validate" or stage is None:
            self.train_dataset = biovid_train
            self.valid_dataset = biovid_val

        if stage == "test" or stage is None:
            self.test_dataset = biovid_test
            

if __name__ == "__main__":

    path_to_frame_labels = "/media/tien/SSD-NOT-OS/pain_intermediate_data/processed_pain_data_no_facedetector/"
    path_to_video_frame = (
        "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/extracted_frame/"
    )
    path_to_3d_latents = (
        "/media/tien/SSD-NOT-OS/pain_intermediate_data/emoca_latent_code/"
    )
    temp_dir = "/media/tien/SSD-NOT-OS/pain_intermediate_data/temp_video"

    # def assert_sample(sample):
    #     # assert sample[0].size(0) == 256
    #     assert sample["au_features"].size(0) == 256
    #     assert sample["emotion_labels"].size(0) == 256
    #     assert sample["temperature_values"].size(0) == 256
    #     assert sample["stimulus_values"].size(0) == 256
    #     assert sample["pspi_no_au43"].size(0) == 256
    #     assert len(sample["latent_3d"]) == 256
    
    from tqdm import tqdm
    import time
    
    num_workers = 4
    
    # arg parser
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--worker_idx', type=int, default=0, help='Dataset split to use')
    args = parser.parse_args()
    
    current_worker = args.worker_idx
    split = args.split
    
    def loop(split="train", temp_dir=None):
        dataset = BioVidDataset(
            path_to_video_frame=path_to_video_frame,
            path_to_frame_labels=path_to_frame_labels,
            path_to_3d_latents=path_to_3d_latents,
            split=split,
            max_length=64,
            is_video=True,
        )
        
        dataset.temp_dir = temp_dir

        print(len(dataset))
        
        start = time.time()

        for i in tqdm(range(0, len(dataset))):
            if i % num_workers == current_worker:
                sample = dataset[i]
            
        print(f"{split} passed:", time.time()-start)
    
    loop(split, temp_dir=temp_dir)
    
    loop(split, temp_dir=temp_dir)

        # assert_sample(sample)

    # assert_sample(dataset[0])
    # assert_sample(dataset[-1])

