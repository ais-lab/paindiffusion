import gradio as gr
import threading
import time
import asyncio
from lightning import Trainer
import numpy as np
from collections import deque

import torch
import yaml
from diffusion.elucidated_for_video import ElucidatedDiffusion

from diffusion.module.utils.biovid import BioVidDM
from inferno_package.render_from_exp import decode_latent_to_image

def load_model(conf_file) -> ElucidatedDiffusion:

    with open(conf_file, "r") as f:
        conf = yaml.safe_load(f)

    best_checkpoint = conf["BEST_CKPT"]
    
    model = ElucidatedDiffusion.from_conf(conf_file)

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        fast_dev_run=1,
        logger=False,
    )

    biovid = BioVidDM.from_conf(conf_file)
    biovid.test_max_length = 64
    trainer.test(model, datamodule=biovid, ckpt_path=best_checkpoint)

    model = model.eval()
    model = model.cuda()
    
    return model

# model = load_model("configure/sample_config.yml")
model = load_model("configure/scale_jawpose_window_32.yml")
default_face = 'default_face/'

# Initialize shared variables
current_stimuli = {
    'pain_stimuli': 30,
    'pain_configuration': 5,
    'emotion_status': 5,
    'scripted_pain_stimuli': None
}

past_frames = None
current_frame = None
stop_threads_flag = False

scheduling_matrix = None

sr = 30  # Sampling rate in Hz
generate_fps = 30  # Frame generation rate in Hz
window_size = 32

frame_queue = deque()
stimuli_queue = deque(maxlen=window_size)  # Fixed size queue

def stimuli_sampling_loop():
    global stop_threads_flag
    while not stop_threads_flag:
        # Sample current stimuli values
        stimuli_sample = current_stimuli.copy()
        # print(f"Stimuli sampling loop: {stimuli_sample['pain_stimuli']}, {stimuli_sample['pain_configuration']}, {stimuli_sample['emotion_status']}")
        # Append to stimuli_queue
        stimuli_queue.append(stimuli_sample)
        # Sleep for sampling interval
        time.sleep(1.0 / sr)

def model_loop():
    global stop_threads_flag
    
    target_interval = 1.0 / generate_fps
    
    while not stop_threads_flag:
        stimuli_values = list(stimuli_queue)

        start = time.time()
        frames = generate_frames(stimuli_values)
        prediction_time = time.time() - start 
        
        frame_interval = prediction_time / len(frames)
        
        # cap the prediction rate to 30fps by sleeping
        if frame_interval < target_interval:
            time.sleep((target_interval - frame_interval)*len(frames))
            frame_interval = target_interval
        
        for frame in frames:
            frame_queue.append((frame, frame_interval))

def generate_frames(stimuli_values):
    
    # construct ctrl tensor
    
    if len(stimuli_values) < window_size:
        return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(window_size)]
    
    emotion_list = [stimuli['emotion_status'] for stimuli in stimuli_values]
    
    pain_config = [stimuli['pain_configuration'] for stimuli in stimuli_values]
    
    pain_stimuli_list = [stimuli['pain_stimuli'] for stimuli in stimuli_values]
    
    [pain_stimuli_list, pain_config, emotion_list] = [torch.tensor(x).float().unsqueeze(0) for x in [pain_stimuli_list, pain_config, emotion_list]]
    
    [pain_stimuli_list, pain_config, emotion_list] = [x.cuda() for x in [pain_stimuli_list, pain_config, emotion_list]]
    
    ctrl = [pain_stimuli_list, pain_config, emotion_list]
        
    # define guide
    
    guide = [0.25, 0.5, 1.0]
    
    global past_frames
    
    prediction_tensor = model.sample_a_chunk(ctrl, guide, past_frames)
    
    past_frames = prediction_tensor.detach().clone()
    
    # print(prediction_tensor.shape)
    
    render_frames = decode_latent_to_image(default_face, prediction_tensor, render=False, save_frame=False)
    
    # scale the frame up to 640x640
    render_frames = [np.array(frame) for frame in render_frames]
    render_frames = [np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1) for frame in render_frames]
    
    return render_frames

def display_loop():
    global current_frame, stop_threads_flag
    while not stop_threads_flag:
        if frame_queue:
            frame, frame_interval = frame_queue.popleft()
            current_frame = frame
            time.sleep(frame_interval)
        else:
            time.sleep(0.01)

def get_frame():
    return current_frame 

# Global variable to keep track of the decay thread
decay_thread = None

def update_pain_stimuli(pain_stimuli):
    current_stimuli['pain_stimuli'] = pain_stimuli

async def decay_pain_stimuli():
    start_value = current_stimuli['pain_stimuli']
    original_value = 30
    duration = 5  # Duration in seconds
    steps = 50
    step_delay = duration / steps
    step_value = (start_value - original_value) / steps

    for _ in range(steps):
        await asyncio.sleep(step_delay)
        start_value -= step_value
        if start_value < original_value:
            start_value = original_value
        current_stimuli['pain_stimuli'] = start_value
        yield start_value  # Update the slider in the UI

def update_other_stimuli(pain_configuration, emotion_status):
    current_stimuli['pain_configuration'] = pain_configuration
    emotion_map = {
        "Anger": 0,
        "Contempt": 1,
        "Disgust": 2,
        "Fear": 3,
        "Happiness": 4,
        "Neutral": 5,
        "Sadness": 6,
        "Surprise": 7
    }
    current_stimuli['emotion_status'] = emotion_map[emotion_status]

# Start threads
stimuli_thread = threading.Thread(target=stimuli_sampling_loop)
model_thread = threading.Thread(target=model_loop)
display_thread = threading.Thread(target=display_loop)
stimuli_thread.start()
model_thread.start()
display_thread.start()

with gr.Blocks() as demo:
    gr.HTML('''
    <h1 class="title is-1 publication-title">PainDiffusion: Can robot express pain?</h1>
    ''')

    with gr.Row():
        pain_stimuli_slider = gr.Slider(
            30, 60, value=30, label="Heat Stimuli", step=0.1, elem_id="pain_stimuli_slider"
        )
        pain_configuration_slider = gr.Slider(5, 11, value=5, label="Pain Configuration", step=0.1)
        emotion_status_radio = gr.Radio(
            choices=[
                "Anger", "Contempt", "Disgust", "Fear",
                "Happiness", "Neutral", "Sadness", "Surprise"
            ],
            value="Neutral",
            label="Emotion Status"
        )

    # Update pain_stimuli in real-time as the slider moves
    pain_stimuli_slider.input(
        fn=update_pain_stimuli,
        inputs=pain_stimuli_slider,
        outputs=None
    )

    # Start decay when the slider is released
    pain_stimuli_slider.release(
        fn=decay_pain_stimuli,
        inputs=None,
        outputs=pain_stimuli_slider  # Update the slider value in the UI
    )

    # Update other stimuli when their sliders change
    pain_configuration_slider.change(
        fn=update_other_stimuli,
        inputs=[pain_configuration_slider, emotion_status_radio],
        outputs=None
    )
    emotion_status_radio.change(
        fn=update_other_stimuli,
        inputs=[pain_configuration_slider, emotion_status_radio],
        outputs=None
    )

    frame_display = gr.Image(label="Current Frame")

    def update_frame():
        while True:
            if current_frame is not None:
                yield current_frame
            time.sleep(0.01)

    demo.load(fn=update_frame, inputs=[], outputs=frame_display)

demo.launch()
