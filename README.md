# PainDiffusion: Can Robots Express Pain?

![Flow 1@1x-25fps](https://github.com/user-attachments/assets/41bf9e82-d544-4ee2-b9e5-bfcf2f7abbe8)

[**Project Page**](https://damtien444.github.io/paindf/) | [**arXiv:2409.11635**](https://arxiv.org/pdf/2409.11635)

Due to the privacy policies of the BioVid Database, we can only release the checkpoints, training code, and inference code. To minimize our effort, we are releasing the training and preprocessing code *as is* for reference purposes. We have only tested and verified the inference code, which includes a Gradio-based demo.

---

## Installation

### Prerequisites
Install [Inferno](https://github.com/radekd91/inferno) for the EMOCA decoder. Follow the instructions [here](https://github.com/damtien444/inferno?tab=readme-ov-file#installation) and download the necessary models for facial reconstruction [here](https://github.com/damtien444/inferno?tab=readme-ov-file#installation). We have slightly modified the original code to generate useful latent variables for the face reconstruction app and to support the `render_from_exp.py` script.

**Note:**  
Installing `pytorch3d` might present compatibility issues due to mismatched versions of CUDA, PyTorch, and `pytorch3d`. If this occurs, install `pytorch3d` separately.

### Setup

```bash
# Create and activate a new Conda environment
conda create python=3.10 -n paindiff 
conda activate paindiff

# Install the required packages and pytorch3d
# Ensure the CUDA version matches your PyTorch installation and system configuration: https://pytorch.org/get-started/locally/

pip install -r requirements.txt

FORCE_CUDA=1 pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

# Clone and set up the Inferno package
git clone https://github.com/damtien444/inferno inferno_package
cd inferno_package/
bash pull_submodules.sh

pip install -e .

# Download the pretrained EMOCA model for PainDiffusion
cd inferno_apps/FaceReconstruction
bash download_assets.sh
```

---

## Running the Online Demo

[**PainDiffusion Demo Video**](https://github.com/user-attachments/assets/d3b130cf-67a9-4064-8961-3bd4516af658)

1. **Download the Model Checkpoint**  
   [**Download Link**](https://drive.google.com/file/d/1sh7JdYWcz-Z-pc30mWtl7TOKxzHwz80V/view?usp=sharing)

2. **Place the Checkpoint**  
   Save the downloaded checkpoint in a directory of your choice.

3. **Update Configuration**  
   Edit `configure/sample_config.yml` and set the `BEST_CKPT` field to the path of your checkpoint.

4. **Run the Demo**  
   ```bash
   python online_run.py
   ```

5. **Access the Demo**  
   Open [http://127.0.0.1:7860/](http://127.0.0.1:7860/) in your web browser.

---

## Troubleshooting: CUDA Version Issues

Installing `pytorch3d` can be tricky, especially if your system has multiple versions of CUDA. Use the following commands to set the appropriate CUDA version for your PyTorch installation:

```bash
# Replace {cuda_version} with the correct version for your system
export CUDA_HOME=/usr/local/cuda-{cuda_version}
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## Acknowledgments

This project is heavily based on the excellent implementations of diffusion models from:  
- [**modular-diffusion**](https://github.com/myscience/modular-diffusion)  
- [**denoising-diffusion-pytorch**](https://github.com/lucidrains/denoising-diffusion-pytorch)  
- [**k-diffusion**](https://github.com/crowsonkb/k-diffusion)  
