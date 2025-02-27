# MI-Eval: Re-thinking Defense Application and Evaluation for Model Inversion

MI-Eval is a collection of tools for researching model inversion attacks and defenses for deep classification models.


## Table of Contents
- [Installation and setup](#installation-and-setup)
- [Features](#features)
- [Usage](#usage)

## Installation and setup

Since this repository relies on other researchers code for defenses and attacks, it clones several other repos into the repository. However, this process has been streamlined using bash scripts. To setup the repo, 

### Steps
1. Clone the repository
2. Setup and activate environment. The repository is devolped to Python 3.11. Eg using conda:
```
conda create -n ENVIRONMENT_NAME python=3.11
```
3. Install dependencies
```
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```
4. Run bash script for setting up other repositories and downloading pretrained GAN's 
```
bash setup_files.bash
```
5. Weights and Biases Logging (Optional, but recommended)


## Features
This section will go present the different defenses, attacks and dataset that this repository will support.

### Defenses
| Name | Citation | Implementation |
|----------|----------|---------|
| TL-DMI     | [Ho et al. 2024](https://arxiv.org/abs/2405.05588) | [Github](https://github.com/hosytuyen/TL-DMI)
| Neg-LS     | [Struppek et al. 2024](https://arxiv.org/abs/2310.06549)   | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks)
| BiDO (WIP) | [Peng et al. 2022](https://arxiv.org/abs/2206.05483)   | [Github](https://github.com/AlanPeng0897/Defend_MI)
| MID (WIP)  | [Wang et al. 2020](https://arxiv.org/abs/2009.05241) | [Github](https://github.com/Jiachen-T-Wang/mi-defense)

### Attacks
| Name | Resolution | Citation | Implementation |
|----------|----------|---------|---------|
| PPA          | High-Res | [Struppek et al. 2022](https://proceedings.mlr.press/v162/struppek22a.html) | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks)
| PLG-MI (WIP) | High-Res | [Yuan et al. 2023](https://arxiv.org/abs/2302.09814)   | [Github](https://github.com/LetheSec/PLG-MI-Attack)
| IF-GMI (WIP) | High-Res | [Qiu et al. 2024](https://arxiv.org/abs/2407.13863) | [Github](https://github.com/final-solution/IF-GMI)
| KED-MI (WIP) | High-Res | [Chen et al. 2021](https://arxiv.org/abs/2010.04092) | [Github](https://github.com/SCccc21/Knowledge-Enriched-DMI)

### Classifiers

We support a wide range 

### Datasets
Most of the datasets are implemented with automatic downloading and processing. 

| Name | Size | Resolution | Downloading |
|----------|----------|---------|---------|
| Facescrub     | x |  High-Res  | Automatic  |
| Standord Dogs | x |  High-Res  | Automatic  |
| CelebA        | x |  High-Res  | Automatic* |
| CIFAR10       | x |  Low-Res   | Automatic  |
| MNIST         | x |  Low-Res   | Automatic  |
| FashionMNIST  | x |  Low-Res   | Automatic  |

(*) There is a bug at the moment with downloading CelebA with gdown (Download img_align_celeba.zip and put in the data/celeba from the link and unzip: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)

This [wandb report](https://api.wandb.ai/links/BreuerLab/rbbr8jqr) contains results from running a Plug-and-Play attack on various model defenses, with some additional insights into the training of target models.

## Usage

### Training Classifiers


### Attacking Classifiers

