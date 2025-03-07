# PainDiffusion: Can Robots Express Pain?

[![Project Page](https://img.shields.io/badge/Project%20Page-blue?logo=github&labelColor=black&link=https%3A%2F%damtien444.github.io%2Fpaindf)](https://damtien444.github.io/paindf/) [![arXiv](https://img.shields.io/badge/arXiv-2409.11635-B31B1B)](https://arxiv.org/pdf/2409.11635)

![Flow 1@1x-25fps](https://github.com/user-attachments/assets/41bf9e82-d544-4ee2-b9e5-bfcf2f7abbe8)

---

Due to the privacy policies of the BioVid Database, we can only release the checkpoints, training code, and inference code. To minimize our effort, we are releasing the training and preprocessing code *as is* for reference purposes. We have only tested and verified the inference code, which includes a Gradio-based demo.

## Installation

### Prerequisites
Install [Inferno](https://github.com/radekd91/inferno) for the EMOCA decoder. Follow the instructions [here](https://github.com/damtien444/inferno?tab=readme-ov-file#installation) and download the necessary models for facial reconstruction [here](https://github.com/damtien444/inferno?tab=readme-ov-file#installation). We have slightly modified the original code to generate useful latent variables for the face reconstruction app and to support the `render_from_exp.py` script.

**Note:**  Installing `pytorch3d` might present compatibility issues due to mismatched versions of CUDA, PyTorch, and `pytorch3d`. If this occurs, install `pytorch3d` separately.

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


## Running the Mesh Demo

Thanks to the Hugging Face Model Hub, you can run PainDiffusion with a single command.

https://github.com/user-attachments/assets/1f5f7a5e-fcaf-4cde-8e5d-01e89d36c230

1. Run the Demo
   ```bash
   python online_run.py
   ```

2. Access at [http://127.0.0.1:7860/](http://127.0.0.1:7860/) in your web browser.

## Driving GaussianAvatars

For a better realistic avatar, we use PainDiffusion to drive [Gaussian Avatars](https://github.com/ShenhanQian/GaussianAvatars/tree/669ee0e428e6dbfa552c63d75df53234c42cfbbd). Follow the step in [this repository](https://github.com/ais-lab/gaussiansp-paindiffusion) to do that.

https://github.com/user-attachments/assets/fafd7913-9ee5-4b55-b506-008ffca51385

---

## Acknowledgments

We thank the previous author for their opensource code. This project is heavily based on the excellent implementations of diffusion models from:  
- [**modular-diffusion**](https://github.com/myscience/modular-diffusion)  
- [**denoising-diffusion-pytorch**](https://github.com/lucidrains/denoising-diffusion-pytorch)  
- [**k-diffusion**](https://github.com/crowsonkb/k-diffusion)  


## Citation

```bibtex

@misc{dam2024paindiffusion,
      title={PainDiffusion: Can robot express pain?}, 
      author={Quang Tien Dam and Tri Tung Nguyen Nguyen and Dinh Tuan Tran and Joo-Ho Lee},
      year={2024},
      eprint={2409.11635},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.11635}, 
}

```
