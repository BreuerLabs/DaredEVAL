*“We don’t manually re-code backprop or rewrite accuracy metrics every time we train a new model, so why are we still re-hacking training data privacy evaluations?”*

Testing whether a model **leaks its training data** is essential to **ML security**. Yet researchers still test vulnerabilities and apply defenses using custom one-off scripts, low-level PyTorch hacking, and crudely aggregated privacy metrics that miss important leaks. This is slow, fragmented, theoretically misaligned, and empirically incomparable across datasets and defenses, and it’s holding back progress in the field.

![Main-GIF](assets/0s-and-1s.gif) ![Main-GIF](assets/0s-and-1s.gif)
## **daredEval**: *A Declarative Paradigm for Model Inversion Evaluation*  
**daredEval** is a new tool that enables us to **concisely and elegantly describe** any defense, **apply it** to any PyTorch model, then **rigorously evaluate** how it leaks training data information **without writing a new ad-hoc codebase each time**.

### **Core Idea: Code Defenses the Way You Reason About Defenses**  
- **Add your PyTorch model** (or pick any standard one);
- **Declare just what your defense modifies** (e.g., how it changes the loss function, adds gradient noise, modifies architecture);
- **Run a single command**:  ```bash python train_classifier.py model=<MODEL> dataset=<DATASET> defense=<MY-NEW-DEFENSE>```
- **daredEval adds your defense takes care of the rest.**  

[comment]: <> (Defenses are implemented as functions that take in an "undefended" PyTorch model and output a new "defended" PyTorch model that inherits from the undefended model. The implementation of this function then amounts to simply overwriting the specific methods of the model that are affected by the defense, isolating the essential features of the defense and saving valuable coding time. The defense is then added to our hierarchical configuration structure using Hydra, so that running the defense can be as simple as)

Under the hood, **daredEval runs structured empirical evaluations** and delivers a **rich and reproducible set of vulnerability measures** that make rigorous leakage comparisons possible. This means:
- Abundant and clear **apples-to-apples evaluations** across models, datasets, and defenses;
- A **unified and consistent way to describe and compare defenses’** essential features;
- R**eproducibility at scale** and **privacy insights that generalize**.

### Table of Contents
- [Features](#features)
- [Installation and 5-Minute Quickstart](#5mqs)
- [More Details](#details)

### Features
ReconKit supports a wide range of model inversion defenses, attacks, target classifiers, and datasets for quick evaluation of SOTA methods:

#### Defenses
| Name | Citation | Implementation | Command (defense=) | 
|----------|----------|---------|---------|
| TL-DMI     | [Ho et al. 2024](https://arxiv.org/abs/2405.05588) | [Github](https://github.com/hosytuyen/TL-DMI) | tldmi
| Neg-LS     | [Struppek et al. 2024](https://arxiv.org/abs/2310.06549)   | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks) | label_smoothing
| RoLSS      | [Hao et al. 2024](https://link.springer.com/chapter/10.1007/978-3-031-73004-7_9) | [Github](https://github.com/Pillowkoh/RoLSS/) | rolss
| Gaussian-Noise | see e.g. [Chaudhuri et al. 2024](https://arxiv.org/abs/2404.02866) | [Github](https://github.com/facebookresearch/hcrbounds) | gaussian_noise
| BiDO       | [Peng et al. 2022](https://arxiv.org/abs/2206.05483)   | [Github](https://github.com/AlanPeng0897/Defend_MI) | bido
| MID        | [Wang et al. 2020](https://arxiv.org/abs/2009.05241) | [Github](https://github.com/Jiachen-T-Wang/mi-defense) | mid

#### Attacks
| Name | Resolution | Paper | Implementation | Command (attack=) | 
|----------|----------|---------|---------| ---------|
| PPA          | High-Res | [Struppek et al. 2022](https://proceedings.mlr.press/v162/struppek22a.html) | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks) | plug_and_play
| PLG-MI * | High-Res | [Yuan et al. 2023](https://arxiv.org/abs/2302.09814)   | [Github](https://github.com/LetheSec/PLG-MI-Attack) | plg_mi
| IF-GMI * | High-Res | [Qiu et al. 2024](https://arxiv.org/abs/2407.13863) | [Github](https://github.com/final-solution/IF-GMI) | if_gmi
| KED-MI * | High-Res | [Chen et al. 2021](https://arxiv.org/abs/2010.04092) | [Github](https://github.com/SCccc21/Knowledge-Enriched-DMI) | ked_mi

\* in development

We thank the authors of the above papers for making their code publicly available.

#### Classifiers

We support a range of pretrained classifiers from torchvision, including the ResNet, DenseNet, ResNeXt, and ResNeSt families, as well as Inceptionv3. We also provide a simple MLP and CNN implementation for low-resolution, small datasets.

A custom classifier can also be tested by implementing `CustomClassifier` in `custom_classifier.py` and running `model=custom` in the command line. Parameters for this can be added easily in `configuration/classifier/custom.yaml`. 


#### Datasets
The datasets are implemented with automatic downloading and processing for ease of use. 

| Name | Size | Resolution | Downloading |
|----------|----------|---------|---------|
| Facescrub     | 141,130 |  High-Res  | Automatic  |
| Stanford Dogs | 20,580  |  High-Res  | Automatic  |
| CelebA        | 202,599 |  High-Res  | Automatic** |
| CIFAR10       | 60.000  |  Low-Res   | Automatic  |
| MNIST         | 60.000  |  Low-Res   | Automatic  |
| FashionMNIST  | 60.000  |  Low-Res   | Automatic  |

** There is a torchvision bug ([link](https://github.com/pytorch/vision/issues/8204#issuecomment-1935737815)) at the moment with downloading CelebA with gdown; this requires the `img_align_celeba.zip` file to be downloaded manually. To do this, download `img_align_celeba.zip` from [this link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg), place the .zip folder in `data/celeba`, and unzip.

#### Attack Metrics
| Name                      | Novel? | Description |
|---------------------------|:--------:|-------------|
| AttackAcc@1              |        | Accuracy of an Inception evaluation model in correctly classifying faces produced by the attack |
| AttackAcc@5              |        | Same as above, with top-5 accuracy |
| Average $\delta_{face}$  |        | Average L2 distance to nearest training data image in the target class, averaged over all target classes, in the FaceNet evaluation model feature space |
| Average $\delta_{eval}$  |        | Same as above, but instead in Inception evaluation model feature space |
| Per-class $\delta_{face}$ | ✅     | Histogram of average L2 distances to nearest training data image in the target class, for each target class, in the FaceNet evaluation model feature space |
| Per-class $\delta_{eval}$ | ✅     | Same as above, but instead in Inception evaluation model feature space |
| FID                      |        | Frechet Inception Distance between Inception feature distributions of training data from the target class and their corresponding reconstructions |
| Knowledge Extraction Score * |    | Classification accuracy of a surrogate model trained on the reconstructed images |

\* in development

### 5 Minute Quick Start <a name="5mqs"></a>

Get started running a wide range of defenses and attacks in minutes, following the steps below.

#### Steps
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

#### Training Classifiers
Train a model with any of our models and datasets easily, such as with the following command:
```
python train_classifier.py dataset=CelebA model=pretrained model.architecture="ResNet152" model.hyper.epochs=50 training.wandb.track=True
```

#### Defending Classifiers
To defend classifiers, simply add the defense name in the command for training classifiers, for example:

```
python train_classifier.py defense=tldmi defense.freeze_layers=7 dataset=FaceScrub model=pretrained model.architecture="ResNet18" training.wandb.track=True
```
All the available configurable parameters and default values for training and defending classifiers can be found in the `.yaml` files located in `configuration/classifier/`.

#### Attacking Classifiers
Attacking classifiers can be configured similarly:
```
python run_model_inversion.py attack=plug_and_play target_wandb_id="TARGET_RUN_ID" attack.evaluation_model.wandb_id="EVAL_RUN_ID" training.wandb.track=True
```
Where `"TARGET_RUN_ID"` can be found in your wandb project. All the available configurable parameters and default values for attacking trained classifiers can be found in the `.yaml` files located in `configuration/model_inversion/`.

### More Details <a name="details"></a>
The repository utilizes [Hydra](https://hydra.cc/docs/intro/) for dynamic hierachical configuration. The default configuration values can be changed in the configuration folder or via the Hydra CLI syntax. The following are part of our Hydra configuration for training classifiers:
* models (e.g. `model=cnn` or `model=pretrained model.architecture=resnet18 model.hyper.lr=0.0001`)
* target model training (e.g. `training.wandb.track=True training.device=cuda`)
* defenses (e.g. `defense=bido defense.a1=0.1 defense.a2=0.05` or `defense=tldmi defense.freeze_layers=6`)

#### Defense Application via Overwriting
Defenses are implemented as functions that take in an "undefended" PyTorch model and output a new "defended" PyTorch model that inherits from the undefended model. The implementation of this function then amounts to simply overwriting the specific methods of the model that are affected by the defense, isolating the essential features of the defense and saving valuable coding time. The defense is then added to our hierarchical configuration structure using Hydra, so that training a model with defense X simply amounts to adding `defense=X` to the command line.




