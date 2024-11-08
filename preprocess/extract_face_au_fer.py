import argparse
import json
import os
import time
from collections import deque
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from batch_face import RetinaFace
from hsemotion.facial_emotions import HSEmotionRecognizer
from MEGraphAU import MEFARG
from MEGraphAU.utils import load_state_dict
from PIL import Image
from sync_modality import (
    convert_frame_id_to_label_id,
    convert_frame_id_to_temperature_id,
)
from torch.nn.functional import interpolate
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
    ]
)

# output_dir = "/media/tien/SSD-NOT-OS/processed_pain_data"

temperature_dir = "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/temperature/"

label_dir = "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/label/"

output_dir = "/media/tien/SSD-NOT-OS/processed_pain_data_no_facedetector/"

diagnose_dir = "/media/tien/SSD-NOT-OS/diagnose_video"

au_len = 12
intensity_values = [deque(maxlen=100) for i in range(au_len)]
temperature_values = deque(maxlen=100)
label_values = deque(maxlen=100)
pspi_values = deque(maxlen=100)

x_data = np.linspace(0, 1, 100)

figure_au = plt.figure(1)
figure_au.suptitle("AU Intensity")

for i in range(100):
    for j in range(au_len):
        intensity_values[j].append(0)

    temperature_values.append(0)
    label_values.append(0)
    pspi_values.append(0)

auIntensityLines = []
for i in range(au_len):
    (_line,) = plt.plot(x_data, intensity_values[i])
    auIntensityLines.append(_line)


figure_temp = plt.figure(2)
figure_temp.suptitle("Temperature")
(temp_line,) = plt.plot(x_data, temperature_values)
figure_temp.gca().set_ylim(25, 75)

figure_label = plt.figure(3)
figure_label.suptitle("Label")
(label_line,) = plt.plot(x_data, label_values)
# figure_label.gca().set_ylim(0, 10)
(pspi_line,) = plt.plot(x_data, label_values)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def update():
    for i in range(au_len):
        auIntensityLines[i].set_ydata(intensity_values[i])

    temp_line.set_ydata(temperature_values)
    label_line.set_ydata(label_values)
    pspi_line.set_ydata(pspi_values)
    figure_au.gca().relim()
    figure_au.gca().autoscale_view()
    figure_label.gca().relim()
    figure_label.gca().autoscale_view()
    # figure_temp.gca().relim()
    # figure_temp.gca().autoscale_view()


def reset():
    for i in range(au_len):
        intensity_values[i].clear()

    temperature_values.clear()
    label_values.clear()

    for i in range(100):
        for j in range(au_len):
            intensity_values[j].append(0)

        temperature_values.append(0)
        label_values.append(0)

    update()


# animation = FuncAnimation(figure, update, interval=200)

# plt.show(block=False)


def save_face(
    frame_buffer,
    crop_face,
    emotions,
    emotion_score,
    au_score,
    video_name,
    no_face_mask,
    display=False,
    save=False,
    temperature_data: pd.DataFrame = None,
    label_data: pd.DataFrame = None,
    au_disfa_score=None,
    use_face_detection=False,
):
    no_face_count = 0

    for idx, (frame, frame_id, video_name) in enumerate(frame_buffer):

        if no_face_mask[idx]:
            no_face_count += 1
            continue

        idx = idx - no_face_count

        if idx < 0:
            continue

        face = crop_face[idx]
        emotion = emotions[idx]

        # turn to list
        _emotion_score = emotion_score[idx].tolist()

        _au_score = au_score[idx].tolist()

        _au_disfa_score = au_disfa_score[idx].tolist()

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

        temp_id = convert_frame_id_to_temperature_id(frame_id)

        label_id = convert_frame_id_to_label_id(frame_id)

        if temp_id > temperature_data.index.max():
            temp_id = temperature_data.index.max()

        if label_id > label_data.index.max():
            label_id = label_data.index.max()

        label = label_data.iloc[label_id]["label"]
        temp_value = temperature_data.iloc[temp_id]["temperature"]

        face_name = f"{frame_id}"

        if display:
            save_dir = os.path.join(diagnose_dir, video_name)
            os.makedirs(save_dir, exist_ok=True)
            # plot emotion
            cv2.putText(
                face,
                emotion,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                5,
            )

            cv2.putText(
                face,
                f"{frame_id}",
                (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                5,
            )

            label_values.append(label)
            temperature_values.append(temp_value)

            for i in range(au_len):
                intensity_values[i].append(_au_score[i])

            pspi_values.append(pspi_no_au43)

            update()
            figure_au.canvas.draw()
            figure_temp.canvas.draw()
            figure_label.canvas.draw()
            img_plot_au = np.array(figure_au.canvas.renderer.buffer_rgba())
            img_plot_temp = np.array(figure_temp.canvas.renderer.buffer_rgba())
            img_plot_label = np.array(figure_label.canvas.renderer.buffer_rgba())
            # show face
            cv2.imshow("face", face)

            cv2.imshow("au", img_plot_au)

            cv2.imshow("temperature", img_plot_temp)

            cv2.imshow("label", img_plot_label)

            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGRA)

            face = cv2.resize(face, (500, 500))

            img_plot_au = cv2.resize(img_plot_au, (face.shape[1], face.shape[0]))
            img_plot_temp = cv2.resize(img_plot_temp, (face.shape[1], face.shape[0]))
            img_plot_label = cv2.resize(img_plot_label, (face.shape[1], face.shape[0]))

            # print(face.shape, img_plot.shape)
            merge = np.concatenate(
                (face, img_plot_au, img_plot_temp, img_plot_label), axis=1
            )

            # save image
            image_path = os.path.join(save_dir, face_name + ".jpg")

            cv2.imwrite(f"{image_path}", merge)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if save:
            save_dir = os.path.join(output_dir, video_name)
            os.makedirs(save_dir, exist_ok=True)

            image_path = os.path.join(save_dir, face_name + ".jpg")
            json_path = os.path.join(save_dir, f"{face_name}.json")

            # no need to save the image anymore because no crop face
            if use_face_detection:
                torchvision.utils.save_image(
                    image_transform(face),
                    image_path,
                )

            face_properties = {
                "emotion": emotion,
                "emotion_score": _emotion_score,
                "au_bp4d_score": _au_score,
                "au_disfa_score": _au_disfa_score,
                "temperature": temp_value,
                "label": label,
                "pspi_no_au43": pspi_no_au43,
            }

            # save face properties as json
            with open(json_path, "w") as f:
                json.dump(face_properties, f, cls=NpEncoder)

        # print(f"Save {face_name} {emotion}")

    # print(no_face_count)


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1] : box[3], box[0] : box[2]]
        out = cv2.resize(
            img, (image_size, image_size), interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1] : box[3], box[0] : box[2]]
        out = (
            imresample(
                img.permute(2, 0, 1).unsqueeze(0).float(), (image_size, image_size)
            )
            .byte()
            .squeeze(0)
            .permute(1, 2, 0)
        )
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def crop_face_with_margin(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)
    return face


def predict(
    frame_buffer: list,
    detector: RetinaFace,
    fer: HSEmotionRecognizer,
    au_bp4d_net: MEFARG,
    au_disfa_net: MEFARG = None,
    use_face_detection=False,
):

    batch_list = [frame for frame, _, _ in frame_buffer]
    no_face_mask = [False for i in range(len(batch_list))]
    if use_face_detection:
        all_faces = detector(batch_list, cv=True)

        crop_face = []

        for batch_idx in range(len(all_faces)):
            # get the first face only
            if len(all_faces[batch_idx]) == 0:
                print(
                    "No face detected",
                )
                no_face_mask[batch_idx] = True

            else:
                all_faces[batch_idx] = sorted(
                    all_faces[batch_idx], key=lambda x: x[0][2], reverse=True
                )

                box = all_faces[batch_idx][0][
                    0
                ]  # batch_id, face_idx, list[box, landmark, score]
                # box to interger
                box = [int(b) for b in box]
                x1, y1, x2, y2 = box

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(0, x2)
                y2 = max(0, y2)

                _face = crop_face_with_margin(
                    batch_list[batch_idx], box, image_size=224, margin=100
                )

                crop_face.append(_face)
        batch_list = crop_face
        if len(crop_face) == 0:
            return [], [], [], [], no_face_mask

    crop_face = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in batch_list]
    emotions, emotion_scores = fer.predict_multi_emotions(crop_face, logits=True)

    batch_list = [image_transform(face) for face in crop_face]

    batch = torch.stack(batch_list)
    batch = batch.to(device=device)

    au_score, _ = au_bp4d_net(batch)
    au_score_disfa, _ = au_disfa_net(batch)

    return crop_face, emotions, emotion_scores, au_score, no_face_mask, au_score_disfa


def extract_face(
    video_frame_dir: str,
    detector: RetinaFace,
    fer: HSEmotionRecognizer,
    au_bp4d_net: MEFARG,
    temperature_data: pd.DataFrame,
    label_data: pd.DataFrame,
    au_disfa_net: MEFARG = None,
    use_face_detection=False,
):
    total_frames = len(os.listdir(video_frame_dir))

    # sort frame by frame id
    video_frame_dir_listdir = sorted(
        os.listdir(video_frame_dir),
        key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1]),
    )

    video_name = os.path.basename(video_frame_dir)

    frame_buffer = []
    max_buffer = 64
    start = time.time()

    for frame_id, frame_name in tqdm(enumerate(video_frame_dir_listdir, start=1)):

        # if frame id already in disk then skip
        if os.path.exists(os.path.join(output_dir, video_name, f"{frame_id}.json")):
            continue

        frame = cv2.imread(os.path.join(video_frame_dir, frame_name))

        frame_buffer.append((frame, frame_id, video_name))

        with torch.no_grad():
            if len(frame_buffer) == max_buffer or frame_id == total_frames - 1:
                # try:
                (
                    crop_face,
                    emotions,
                    emotion_scores,
                    au_score,
                    no_face_mask,
                    au_disfa_score,
                ) = predict(
                    frame_buffer,
                    detector,
                    fer,
                    au_bp4d_net,
                    au_disfa_net,
                    use_face_detection=use_face_detection,
                )
                # print(emotions)
                save_face(
                    frame_buffer,
                    crop_face,
                    emotions,
                    emotion_scores,
                    au_score,
                    video_name,
                    no_face_mask,
                    save=False,
                    display=True,
                    temperature_data=temperature_data,
                    label_data=label_data,
                    au_disfa_score=au_disfa_score,
                    use_face_detection=use_face_detection,
                )

                frame_buffer = []

                # # print fps
                # if frame_id % 100 == 0:
                #     print(f"FPS: {100/(time.time()-start)}")
                #     start = time.time()


if __name__ == "__main__":
    video_frame_dir = glob(
        "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/extracted_frame/*"
    )

    # arg parser
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--video_files_split", type=str, required=False, default="2")

    # arg = parser.parse_args()

    # if arg.video_files_split == "1":
    #     video_frame_dir = video_frame_dir[: len(video_frame_dir) // 2]
    # elif arg.video_files_split == "2":
    #     video_frame_dir = video_frame_dir[len(video_frame_dir) // 2 :]
    # else:
    #     video_frame_dir = video_frame_dir

    # print(f"Split video files: {arg.video_files_split}")
    # print(f"Total video files: {len(video_frame_dir)}")

    emotion_model_name = "enet_b2_8"
    fer = HSEmotionRecognizer(model_name=emotion_model_name, device=device)

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

    detector = RetinaFace(gpu_id=0, network="mobilenet")

    # write to file the video has been processed
    # write_file = f"processed_{arg.video_files_split}.txt"

    # ignore_list = [
    #     "082315_w_60",
    #     "082414_m_64",
    #     "082909_m_47",
    #     "083009_w_42",
    #     "083013_w_47",
    #     "083109_m_60",
    #     "083114_w_55",
    #     "091914_m_46",
    #     "092009_m_54",
    #     "092014_m_56",
    #     "092509_w_51",
    #     "092714_m_64",
    #     "100514_w_51",
    #     "100914_m_39",
    #     "101114_w_37",
    #     "101209_w_61",
    #     "101809_m_59",
    #     "101916_m_40",
    #     "111313_m_64",
    #     "120614_w_61",
    # ]

    for video in tqdm(video_frame_dir):
        print(video)
        if "081609_w_40" not in video:
            continue

        # if os.path.basename(video) in ignore_list:
        #     continue

        temperature_data = os.path.join(
            temperature_dir, os.path.basename(video) + ".csv"
        )

        label_data = os.path.join(label_dir, os.path.basename(video) + ".csv")

        # parse the temperature data from csv
        temperature_data = pd.read_csv(temperature_data, delimiter="\t")

        label_data = pd.read_csv(label_data, delimiter="\t")

        # print(temperature_data.iloc[0])

        # if video already processed then skip
        # if os.path.exists(write_file):
        #     with open(write_file, "r") as f:
        #         processed_videos = f.readlines()
        #         processed_videos = [video.strip() for video in processed_videos]
        #         print(processed_videos)
        #         if os.path.basename(video) in processed_videos:
        #             continue

        video_name = os.path.basename(video)
        extract_face(
            video,
            detector,
            fer,
            au_bp4d_net,
            temperature_data,
            label_data,
            au_disfa_net,
            use_face_detection=False,
        )
        # with open(write_file, "a") as f:
        #     f.write(video_name + "\n")
        reset()
