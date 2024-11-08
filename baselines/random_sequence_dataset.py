import random

from diffusion.module.utils.biovid import BioVidDataset 

path_to_frame_labels = "/media/tien/SSD-NOT-OS/pain_intermediate_data/processed_pain_data_no_facedetector/"
path_to_video_frame = (
    "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/extracted_frame/"
)
path_to_3d_latents = (
    "/media/tien/SSD-NOT-OS/pain_intermediate_data/emoca_latent_code/"
)
temp_dir = "/media/tien/SSD-NOT-OS/pain_intermediate_data/temp_video"

dataset = BioVidDataset(
    path_to_video_frame=path_to_video_frame,
    path_to_frame_labels=path_to_frame_labels,
    path_to_3d_latents=path_to_3d_latents,
    split="train",
    max_length=608,
    is_video=True,
)


def random_sequence():
    rndix = random.randint(0, len(dataset) - 1)
    return dataset[rndix]
