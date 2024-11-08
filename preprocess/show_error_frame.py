import os
from glob import glob

import torchvision
from extract_face_au_fer import predict, save_face
from PIL import Image

original = "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/extracted_frame/"

processed = "/media/tien/SSD-NOT-OS/fix_processed_pain_data/"

# check if the processed frame id is the same as the original frame id
# if not print the name of the video and the id of the missing frame
missing_frame = {}

for video in os.listdir(original):

    # start from 1
    original_frame = glob(original + video + "/*.jpg")

    # set the processed frame id to be the same as the original frame id
    original_frame_id = [frame.split("/")[-1].split(".")[0] for frame in original_frame]
    original_frame_id = set(original_frame_id)

    # start from 1
    processed_frame = glob(processed + video + "/*.jpg")
    processed_frame_id = [
        frame.split("/")[-1].split(".")[0] for frame in processed_frame
    ]
    processed_frame_id = set(processed_frame_id)

    if len(original_frame) != len(processed_frame):
        print(
            video,
            len(original_frame),
            len(processed_frame),
            len(original_frame) - len(processed_frame),
        )
        print("Missing frame id: ", original_frame_id - processed_frame_id)
        print(
            "Drop rate: ",
            (len(original_frame_id) - len(processed_frame_id)) / len(original_frame_id),
        )

    # get the full path of the missing frame
    missing_frame[video] = original_frame_id - processed_frame_id


import torch
from batch_face import RetinaFace
from hsemotion.facial_emotions import HSEmotionRecognizer
from MEGraphAU import MEFARG
from MEGraphAU.utils import load_state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_model_name = "enet_b2_8"
fer = HSEmotionRecognizer(model_name=emotion_model_name, device=device)

au = MEFARG(num_classes=12, backbone="swin_transformer_base")
au_path = "/home/tien/fr2-pain/preprocess/MEGraphAU/checkpoints/ME-GraphAU_swin_base_BP4D/MEFARG_swin_base_BP4D_fold1.pth"
au_net = load_state_dict(au, au_path)
au_net = au_net.to(device=device)
au_net.eval()

detector = RetinaFace(gpu_id=0, network="resnet50")
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
    ]
)


import cv2
from torchvision import transforms

transform = transforms.ToTensor()


frame_buffer = []
max_buffer = 5

for video, missing_frame_set in missing_frame.items():

    # create path from the original frame of the missing frame for later extraction
    for rel_id, frame_id in enumerate(missing_frame_set):
        # print('"',original + video + "/frame_{}.jpg".format(frame_id+1), '"', sep="")
        # print('"',processed + video + "/{}.jpg".format(frame_id), '"', sep="")

        image = cv2.imread(original + video + "/{}.jpg".format(frame_id))
        # cv2.imshow("face", image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)

        frame_buffer.append((image, frame_id, video))

        if len(frame_buffer) == max_buffer or rel_id == len(missing_frame_set) - 1:

            images = [frame[0] for frame in frame_buffer]

            all_faces = detector.detect(images, cv=True)

            crop_face = []

            no_face_mask = [False for _ in range(len(all_faces))]

            for batch_idx in range(len(all_faces)):
                # get the first face only
                if len(all_faces[batch_idx]) == 0:
                    print(
                        "No face detected",
                    )
                    no_face_mask[batch_idx] = True

                else:

                    # sort all faces by score
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

                    crop_face.append(images[batch_idx][y1:y2, x1:x2])

                    # cv2.imshow(f"crop", images[batch_idx][y1:y2, x1:x2])

                    # if cv2.waitKey(0) & 0xFF == ord("q"):
                    #     break

            if len(crop_face) == 0:
                frame_buffer = []
                continue

            crop_face = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in crop_face]
            emotions, emotion_scores = fer.predict_multi_emotions(
                crop_face, logits=True
            )

            batch_list = [image_transform(face) for face in crop_face]

            batch = torch.stack(batch_list)
            batch = batch.to(device=device)

            au_score, _ = au_net(batch)

            # save_face(frame_buffer, crop_face, emotions, emotion_scores, au_score, video, no_face_mask, save=True)

            frame_buffer = []

            # show the crop face
            # for face in crop_face:
            #     cv2.imshow("crop face", face)

            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break
