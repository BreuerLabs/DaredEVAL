# MI-Eval: Re-thinking Defense Application and Evaluation for Model Inversion

MI-Eval is a collection of tools for researching model inversion attacks and defenses for deep facial recognition models.


## Table of Contents
- [Installation and setup](#installation-and-setup)
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








## Usage




## Defenses
| Name | Citation | Implementation |
|----------|----------|---------|
| TL-DMI    | [Ho et al. 2024](https://arxiv.org/abs/2405.05588) | [Github](https://github.com/hosytuyen/TL-DMI)
| Neg-LS    | [Struppek et al. 2024](https://arxiv.org/abs/2310.06549)   | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks)
| BiDO (WIP)   | [Peng et al. 2022](https://arxiv.org/abs/2206.05483)   | [Github](https://github.com/AlanPeng0897/Defend_MI)
| MID (WIP)   | [Wang et al. 2020](https://arxiv.org/abs/2009.05241) | [Github](https://github.com/Jiachen-T-Wang/mi-defense)

## Attacks
| Name | Citation | Implementation |
|----------|----------|---------|
| PPA    | [Struppek et al. 2022](https://proceedings.mlr.press/v162/struppek22a.html) | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks)
| PLG-MI (WIP)   | [Yuan et al. 2023](https://arxiv.org/abs/2302.09814)   | [Github](https://github.com/LetheSec/PLG-MI-Attack)
| IF-GMI (WIP)   | [Qiu et al. 2024](https://arxiv.org/abs/2407.13863) | [Github](https://github.com/final-solution/IF-GMI)
| KED-MI (WIP)  | [Chen et al. 2021](https://arxiv.org/abs/2010.04092) | [Github](https://github.com/SCccc21/Knowledge-Enriched-DMI)

This [wandb report](https://api.wandb.ai/links/BreuerLab/rbbr8jqr) contains results from running a Plug-and-Play attack on various model defenses, with some additional insights into the training of target models.
