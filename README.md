# PainDiffusion: Can robot express pain?

[Screencast from 11-21-2024 03:40:59 PM.webm](https://github.com/user-attachments/assets/d3b130cf-67a9-4064-8961-3bd4516af658)

[Project page](https://damtien444.github.io/paindf/) | [arXiv:2409.11635](https://arxiv.org/pdf/2409.11635)

Due to the privacy policies of the Biovid Database, we can only release the checkpoints, training code, and inference code. To save our effort, we are going to release the training code and preprocess code as is for reference purposes. We only test and make sure the inference code is runnable with a Gradio demo. 

## Installation

Install [inferno](https://github.com/radekd91/inferno) for the EMOCA decoder. Follow the instructions [here](https://github.com/damtien444/inferno?tab=readme-ov-file#installation), then follow [this](https://github.com/damtien444/inferno?tab=readme-ov-file#installation) to download the necessary models for facial reconstruction. We slightly modified the original code to generate useful latent in the face reconstruction app and to serve the render_from_exp.py script.

There might be problems with installing pytorch3d, which may come from mismatched versions of CUDA, PyTorch, and pytorch3d. Please separately install pytorch3d if there are problems with installing it.

```bash

conda create python=3.10 -n paindiff 
conda activate paindiff

# Install environment and pytorch3d
# please be mindful that the cuda version should be matched for pytorch and your current cuda system, https://pytorch.org/get-started/locally/

pip install -r requirements.txt

FORCE_CUDA=1 pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

git clone https://github.com/damtien444/inferno inferno_package
cd inferno_package/
bash pull_submodules.sh

pip install -e .

# Download the pretrainpaindiffed EMOCA
cd inferno_apps/FaceReconstruction
bash download_assets.sh

# back to paindiffusion folder and install requirements
cd ../../..
pip install -r requirements.txt
```

# Run the online demo


1. **Download the model checkpoint**: [Download Link](https://drive.google.com/file/d/1sh7JdYWcz-Z-pc30mWtl7TOKxzHwz80V/view?usp=sharing).
2. **Place the checkpoint**: Put the downloaded checkpoint in a location of your choice.
3. **Update configuration**: Edit `configure/sample_config.yml` and set the `BEST_CKPT` path to the location of your checkpoint.
4. **Run the demo**:

   ```bash
   python online_run.py
   ```

5. **Access the demo**: Open [http://127.0.0.1:7860/](http://127.0.0.1:7860/) in your web browser.

![Flow 1@1x-25fps](https://github.com/user-attachments/assets/41bf9e82-d544-4ee2-b9e5-bfcf2f7abbe8)

# Maybe this cuda trick can help you

It's non-trivial to install pytorch3d. I have been stuck at this step many times trying to reproduce the environment. If your system has more than one CUDA version, then use the following set of commands to set your CUDA version that is appropriate for the one used for PyTorch.
```bash
set -x CUDA_HOME /usr/local/cuda-{cuda_version}
set -x CUDA_PATH $CUDA_HOME
set -x PATH $CUDA_HOME/bin $PATH
set -x LD_LIBRARY_PATH $CUDA_HOME/lib64 $LD_LIBRARY_PATH
```


## Thanks
This project is heavily based on the beautiful implementation of diffusion models: [modular-diffusion](https://github.com/myscience/modular-diffusion), [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [k-diffusion](https://github.com/crowsonkb/k-diffusion)
