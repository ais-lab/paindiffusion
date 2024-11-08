# PainDiffusion: Can robot express pain?

![Flow 1@1x-25fps](https://github.com/user-attachments/assets/41bf9e82-d544-4ee2-b9e5-bfcf2f7abbe8)

[Project page](https://damtien444.github.io/paindf/) | [arXiv:2409.11635](https://arxiv.org/pdf/2409.11635)


Due to the privacy policies of the Biovid Database, we can only release the checkpoints, training code, and inference code. To save our effort, we are going to release the training code and preprocess code as is for reference purposes. We only test and make sure the inference code is runnable with a Gradio demo. 

## Installation

Install [inferno](https://github.com/radekd91/inferno) for the EMOCA decoder. Follow the instructions [here](https://github.com/damtien444/inferno?tab=readme-ov-file#installation), then follow [this](https://github.com/damtien444/inferno?tab=readme-ov-file#installation) to download the necessary models for facial reconstruction. We slightly modified the original code to generate useful latent in the face reconstruction app and to serve the render_from_exp.py script.

There might be problems with installing pytorch3d, which may come from mismatched versions of CUDA, PyTorch, and pytorch3d. Please separately install pytorch3d if there are problems with installing it.

```bash

git clone https://github.com/damtien444/inferno 
cd inferno/
bash pull_submodules.sh

conda create python=3.10 -n paindiff 
conda activate paindiff

# Install pytorch and pytorch3d
# please be mindful that the cuda version should be matched for pytorch and your current cuda system, https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
FORCE_CUDA=1 pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
conda env update --name paindiff --file conda-environment_py39_cu12_torch2.yaml

pip install -e .

# Download the pretrained EMOCA
cd inferno_apps/FaceReconstruction
bash download_assets.sh

# back to paindiffusion folder and install requirements
cd ../../..
pip install -r requirements.txt
```



## Thanks
This project is heavily based on the beautiful implementation of diffusion models: [modular-diffusion](https://github.com/myscience/modular-diffusion), [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [k-diffusion](https://github.com/crowsonkb/k-diffusion)