#!/bin/bash


export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

python extract_face_au_fer.py --video_files_split 1