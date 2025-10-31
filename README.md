# 🧬 Enhancing Microalgae Detection through Diffusion-Based Data Augmentation

**Official Implementation for Image Generation of Microalgae Using Diffusion Models**  
(Submitted to *The Visual Computer*, 2025)

---

###  Overview

This repository contains the official implementation of **DDEDA**, a *Diffusion-Driven Data Enhancement* framework designed to generate realistic microalgae images for data augmentation. The framework uses diffusion models to create diverse and high-quality microalgae images, addressing the issue of data scarcity in the field of biological image analysis. This can be used for training detection models in environments where labeled data is limited.

In this repository, you will find code for training the diffusion model on microalgae datasets and for generating new samples. This will enable you to expand datasets for downstream tasks like image classification or object detection.

---

###  Requirements
```bash
torch==2.0
diffusers==0.11
opencv-python==4.5.3
tqdm==4.64.0
numpy==1.21.2
matplotlib==3.5.1
```



###  Dataset Preparation
To train the diffusion model, you need a dataset of cropped microalgae instances.





###  Training the Diffusion Model
To train the diffusion model, run the following command:
```bash
python train.py
```


###  Dataset
VisAlgae2023:https://tianchi.aliyun.com/dataset/169283

VisAlgae2022:https://tianchi.aliyun.com/competition/entrance/532036/information















###  Environment Setup

Follow the instructions below to set up the environment and dependencies:

```bash
# Clone the repository
git clone https://github.com/ksazs/DDEDA.git
cd DDEDA

# Create a conda environment
conda create -n ddeda python=3.10 -y
conda activate ddeda

# Install the required dependencies
pip install -r requirements.txt
