
import os
import faiss

from tqdm import tqdm

from diffusion.module.utils.biovid import BioVidDataset 

from faiss import write_index, read_index



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

if not os.path.exists("large.index"):
    index = faiss.IndexFlatL2(608)
    
    print("building index for nearest neighbor")
    for sample in tqdm(dataset):
        index.add(x=sample['ctrl'][2].unsqueeze(0).numpy())
    
    write_index(index, "large.index")

else:
    index = read_index("large.index")


    
def nearest_neighbor(sample):
    D, I = index.search(sample['ctrl'][2].unsqueeze(0).numpy(), k=1)
    return dataset[I.item()]