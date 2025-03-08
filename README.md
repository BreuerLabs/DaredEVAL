# MI-Eval: Re-thinking Defense Application and Evaluation for Model Inversion

MI-Eval is a collection of tools for researching model inversion attacks and defenses for deep classification models.

### Disclaimer
This repository is still in alpha. Features that are still in development will be marked by `*`.

## Table of Contents
- [Installation and setup](#installation-and-setup)
- [Features](#features)
- [Usage](#usage)

## Installation and setup

This repository uses original authors' code where possible for defense and attack implementations; as such, it clones several other repos into the repository. However, this process has been streamlined using bash scripts.

### Steps
1. Clone the repository.
2. Setup and activate your Python environment. The repository is developed for use with Python 3.11. For example, using conda:
```
conda create -n ENVIRONMENT_NAME python=3.11
```
3. Install dependencies.
```
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```
4. Run bash script for setting up other repositories and downloading pretrained GAN's.
```
bash setup_files.bash
```
5. Weights and Biases Logging (Optional, but recommended).
   - Setup a [Weights and Biases](https://wandb.ai/site/) account
   - Create a project
   - Create a [wandb API key](https://docs.wandb.ai/quickstart/), make a file called `secret.txt` in the main scope of the repository, and paste the API key there.
   - Set the entity and project names in the configuration (`configuration/classifier/training/default.yaml` and `configuration/model_inversion/training/default.yaml` respectively)
   - Run scripts with `training.wandb.track=True` in the command line, or set it as default in the configuration.
   - Enjoy smart and scalable cloud logging for training classifiers and model inversion. 🚀 

## Features
This codebase supports a wide range of model inversion defenses, attacks, and datasets. To see some examples, this [wandb report](https://api.wandb.ai/links/BreuerLab/rbbr8jqr) contains results from running a Plug-and-Play attack on various model defenses, with some additional insights into the training of target models.

### Defenses
| Name | Citation | Implementation | Command (defense=) | 
|----------|----------|---------|---------|
| TL-DMI     | [Ho et al. 2024](https://arxiv.org/abs/2405.05588) | [Github](https://github.com/hosytuyen/TL-DMI) | tldmi
| Neg-LS     | [Struppek et al. 2024](https://arxiv.org/abs/2310.06549)   | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks) | label_smoothing
| RoLSS      | [Hao et al. 2024](https://link.springer.com/chapter/10.1007/978-3-031-73004-7_9) | [Github](https://github.com/Pillowkoh/RoLSS/) | rolss
| BiDO       | [Peng et al. 2022](https://arxiv.org/abs/2206.05483)   | [Github](https://github.com/AlanPeng0897/Defend_MI) | bido
| MID        | [Wang et al. 2020](https://arxiv.org/abs/2009.05241) | [Github](https://github.com/Jiachen-T-Wang/mi-defense) | mid

### Attacks
| Name | Resolution | Citation | Implementation | Command (attack=) | 
|----------|----------|---------|---------| ---------|
| PPA          | High-Res | [Struppek et al. 2022](https://proceedings.mlr.press/v162/struppek22a.html) | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks) | plug_and_play
| PLG-MI * | High-Res | [Yuan et al. 2023](https://arxiv.org/abs/2302.09814)   | [Github](https://github.com/LetheSec/PLG-MI-Attack) | plg_mi
| IF-GMI * | High-Res | [Qiu et al. 2024](https://arxiv.org/abs/2407.13863) | [Github](https://github.com/final-solution/IF-GMI) | if_gmi
| KED-MI * | High-Res | [Chen et al. 2021](https://arxiv.org/abs/2010.04092) | [Github](https://github.com/SCccc21/Knowledge-Enriched-DMI) | ked_mi

We thank the authors of the above papers for making their code publicly available.

### Classifiers

We support a simple MLP and CNN implementation, as well as a range of pretrained classifiers from torchvision, including ResNet18, ResNet152, and Inceptionv3. 

A custom classifier or defenses can also be tested by implementing `CustomClassifier` in `custom_classifier.py` and running `model=custom` in the command line. Parameters for this can be added easily in `configuration/classifier/custom.yaml`. 


### Datasets
Most of the datasets are implemented with automatic downloading and processing. 

| Name | Size | Resolution | Downloading |
|----------|----------|---------|---------|
| Facescrub     | 141,130 |  High-Res  | Automatic  |
| Stanford Dogs | 20,580  |  High-Res  | Automatic  |
| CelebA        | 202,599 |  High-Res  | Automatic* |
| CIFAR10       | 60.000  |  Low-Res   | Automatic  |
| MNIST         | 60.000  |  Low-Res   | Automatic  |
| FashionMNIST  | 60.000  |  Low-Res   | Automatic  |

(*) There is a torchvision bug (link?) at the moment with downloading CelebA with gdown; this requires the `img_align_celeba.zip` file to be downloaded manually. To do this, download `img_align_celeba.zip` from [this link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg), place the .zip folder in `data/celeba`, and unzip.

## Usage
The repository utilizes [Hydra](https://hydra.cc/docs/intro/) for dynamic hierachical configuration. The default configuration values can be changed in the configuration folder or via the Hydra CLI syntax.

### Training Classifiers
It easy to shift between datasets and models in the command line as such:
```
python train_classifier.py dataset=CelebA model=pretrained model.architecture="ResNet152" model.hyper.epochs=50 training.wandb.track=True
```

### Defending Classifiers
To defend classifiers, simply add the defense in the command for training classifiers:

```
python train_classifier.py defense=tldmi dataset=CelebA model=pretrained model.architecture="ResNet18" training.wandb.track=True
```

### Attacking Classifiers
Attacking classifiers can be done configured similarly:
```
python run_model_inversion.py dataset=FaceScrub attack=plug_and_play target_wandb_id="TARGET_RUN_ID" attack.evaluation_model.wandb_id="EVAL_RUN_ID" training.wandb.track=True
```
