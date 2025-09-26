<div align="center">

# PainDiffusion: Learning to Express Pain
<p>
🦾🔥 IROS 2025 Best Paper Award Finalist 🦾🔥
</p>
<!-- Badges -->
<p>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/ais-lab/paindiffusion" alt="last update" />
  </a>
  <a href="https://damtien444.github.io/paindf/">
    <img src="https://img.shields.io/badge/Project%20Website-blue?logo=github&labelColor=black&link=https%3A%2F%damtien444.github.io%2Fpaindf" alt="homepage" />
  </a>
  <a href="https://arxiv.org/pdf/2409.11635">
    <img src="https://img.shields.io/badge/arXiv-2409.11635-B31B1B" alt="arxiv" />
  </a>
  <a href="https://github.com/ais-lab/paindiffusion/stargazers">
    <img src="https://img.shields.io/github/stars/ais-lab/paindiffusion" alt="stars" />
  </a>
</p>


</div>

---

<div align="center">
   
![Flow 1@1x-25fps](https://github.com/user-attachments/assets/41bf9e82-d544-4ee2-b9e5-bfcf2f7abbe8)

</div>

## Introduction

We introduce a generative model intended for robotic facial expression. It can generate expressions according to a signal of pain stimuli, continuously, without divergence. You can install and run it easily with the [instructions](###instructions) provided here.

Q&A:

- Is it big? -> It is small (5.4M parameters).
- How can we know it is not divergent? -> We let it run indefinitely. We compute metrics on longer sequences (on the test set).

Code release: Due to the privacy policies of the BioVid Database, we can only release the checkpoints, training code, and inference code. To minimize our effort, we are releasing the training and preprocessing code *as is* for reference purposes. We have only tested and verified the inference code, which includes a Gradio-based demo.

## Installation

### Prerequisites
Install [Inferno](https://github.com/radekd91/inferno) for the EMOCA decoder. Follow the instructions [here](https://github.com/damtien444/inferno?tab=readme-ov-file#installation) and download the necessary models for facial reconstruction [here](https://github.com/damtien444/inferno?tab=readme-ov-file#installation). We have slightly modified the original code to generate useful latent variables for the face reconstruction app and to support the `render_from_exp.py` script. **Or you can just simply follow the instructions below.**

**Note:**  Installing `pytorch3d` might present compatibility issues due to mismatched versions of CUDA, PyTorch, and `pytorch3d`. Please confirm your version of CUDA and the version that PyTorch compiled with.

### Instructions

```bash
# Create and activate a new Conda environment
conda create python=3.10 -n paindiff 
conda activate paindiff
pip install -r requirements.txt

# Install the required packages and pytorch3d
# Ensure the CUDA version matches your PyTorch installation and system configuration: https://pytorch.org/get-started/locally/
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake  # use the right CUDA version that you saw when run the requirement installation
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX  # for compilation

# Reactivate the environment for new env config
conda deactivate
conda activate paindiff

# Install pytorch3d with cuda support
FORCE_CUDA=1 pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

# Clone and set up the Inferno package
git clone https://github.com/damtien444/inferno inferno_package
cd inferno_package/
bash pull_submodules.sh
pip install -e .

# Download the pretrained EMOCA model for PainDiffusion
cd inferno_apps/FaceReconstruction
bash download_assets.sh

# Return to the repo's root dir
cd ../../..
```


## Running the Mesh Demo

Thanks to the Hugging Face Model Hub, you can run PainDiffusion with a single command.

```bash
python online_run.py
```
Access at [http://127.0.0.1:7860/](http://127.0.0.1:7860/) in your web browser.

https://github.com/user-attachments/assets/1f5f7a5e-fcaf-4cde-8e5d-01e89d36c230

## Driving GaussianAvatars

For a better, realistic avatar, we use PainDiffusion to drive [Gaussian Avatars](https://github.com/ShenhanQian/GaussianAvatars/tree/669ee0e428e6dbfa552c63d75df53234c42cfbbd). Follow the steps in [this repository](https://github.com/ais-lab/gaussiansp-paindiffusion) to do that.

https://github.com/user-attachments/assets/fafd7913-9ee5-4b55-b506-008ffca51385

---

## Acknowledgments

We thank the previous author for their open-source code. This project is heavily based on the excellent implementations of diffusion models from:  
- [**modular-diffusion**](https://github.com/myscience/modular-diffusion)  
- [**denoising-diffusion-pytorch**](https://github.com/lucidrains/denoising-diffusion-pytorch)  
- [**k-diffusion**](https://github.com/crowsonkb/k-diffusion)  


## Citation

```bibtex

@misc{dam2025paindiffusionlearningexpresspain,
      title={PainDiffusion: Learning to Express Pain}, 
      author={Quang Tien Dam and Tri Tung Nguyen Nguyen and Yuki Endo and Dinh Tuan Tran and Joo-Ho Lee},
      year={2025},
      url={https://arxiv.org/abs/2409.11635}, 
}

```
