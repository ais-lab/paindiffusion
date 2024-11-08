import argparse
import json
import os
from glob import glob

import torch
from inferno.datasets.ImageTestDataset import TestData
from inferno_apps.FaceReconstruction.utils.load import load_model
from tqdm import tqdm

path_to_model = "/home/tien/inferno/assets/" + "FaceReconstruction/models"
model_name = "EMICA-CVT_flame2020_notexture"
face_rec_model, conf = load_model(path_to_model, model_name)
face_rec_model.cuda()
face_rec_model.eval()

video_frame_dir = "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/extracted_frame"
output_dir = "/media/tien/SSD-NOT-OS/pain_intermediate_data/emoca_latent_code"

video_frame_dir = glob(video_frame_dir + "/*")

parser = argparse.ArgumentParser()
parser.add_argument("--video_files_split", type=str, required=False, default="2")

arg = parser.parse_args()

if arg.video_files_split == "1":
    video_frame_dir = video_frame_dir[: len(video_frame_dir) // 2]
elif arg.video_files_split == "2":
    video_frame_dir = video_frame_dir[len(video_frame_dir) // 2 :]
else:
    video_frame_dir = video_frame_dir

print(f"Processing {len(video_frame_dir)} videos")

for video in tqdm(video_frame_dir):
    _video_frame_dir = video
    video = os.path.basename(video)
    os.makedirs(os.path.join(output_dir, video), exist_ok=True)

    frame_path_list = os.listdir(_video_frame_dir)
    # print(frame_path_list)
    print(video)
    frame_path_list = [frame.split(".")[0].split("_")[1] for frame in frame_path_list]

    output_frame_path_list = os.listdir(os.path.join(output_dir, video))
    output_frame_path_list = [frame.split(".")[0] for frame in output_frame_path_list]

    print(len(frame_path_list), len(output_frame_path_list))

    if len(set(frame_path_list) - set(output_frame_path_list)) == 0:
        print(f"Skipping {video}")
        continue
    else:
        print(
            f"Processing {video} len {len(set(frame_path_list) - set(output_frame_path_list))}"
        )
        frame_path_list = list(set(frame_path_list) - set(output_frame_path_list))
        frame_path_list = [
            os.path.join(_video_frame_dir, "frame_" + frame + ".jpg")
            for frame in frame_path_list
        ]

    video_dataset = TestData(frame_path_list, face_detector="fan", max_detection=1)
    dataloader = torch.utils.data.DataLoader(
        video_dataset, batch_size=16, shuffle=False, num_workers=0
    )

    for batch in tqdm(dataloader):

        image_paths = batch["image_path"][0]
        image_ids = [
            os.path.basename(image_path).split(".")[0].split("_")[1]
            for image_path in image_paths
        ]

        batch["image"] = batch["image"].cuda()

        # print(image_paths)

        latent_code, ring_size = face_rec_model.extract_latents(
            batch, training=False, validation=False
        )

        # 1, 300
        shapecode = latent_code["shapecode"]

        # bs, 50
        texcode = latent_code["texcode"]

        # bs, 100
        expcode = latent_code["expcode"]

        # bs, 3
        jawpose = latent_code["jawpose"]

        # bs, 3
        globalpose = latent_code["globalpose"]

        # bs, 3
        cam = latent_code["cam"]

        # bs, 27
        lightcode = latent_code["lightcode"]

        bs = len(image_paths)
        for image_idx in range(bs):
            output_json = {
                "texcode": texcode[image_idx].tolist(),
                "expcode": expcode[image_idx].tolist(),
                "jawpose": jawpose[image_idx].tolist(),
                "globalpose": globalpose[image_idx].tolist(),
                "cam": cam[image_idx].tolist(),
                "lightcode": lightcode[image_idx].tolist(),
            }

            # save json
            output_json_path = os.path.join(
                output_dir, video, f"{image_ids[image_idx]}.json"
            )
            # print(output_json_path)
            with open(output_json_path, "w") as f:
                json.dump(output_json, f)
                # print(f"Saved {output_json_path}")

    #     break
    # break
