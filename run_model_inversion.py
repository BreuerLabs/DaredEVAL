from omegaconf import OmegaConf
import hydra
import wandb
import os
import torch
import sys
import time

from classifiers.get_model import get_model
from defenses.get_defense import get_defense
from data_processing.data_loaders import get_data_loaders
from data_processing.datasets import get_datasets
from data_processing.data_augmentation import get_transforms

from Plug_and_Play_Attacks.utils.attack_config_parser import AttackConfigParser
from model_inversion.plug_and_play.attack import attack
from model_inversion.plug_and_play.modify_to_pnp_repo import model_compatibility_wrapper, convert_configs


from utils import wandb_helpers, load_trained_models

@hydra.main(config_path="configuration/model_inversion", config_name="config.yaml", version_base="1.3")
def run_model_inversion(attack_config):

    
    if torch.cuda.is_available() and attack_config.training.device != "cuda":
        question = f"\nCuda is available but not configured from command to be used! Do you wish to use cuda instead of {attack_config.training.device}?\nType y to use cuda, enter if not:"
        
        use_cuda = input(question)
        
        if use_cuda.lower().strip() == "y":
            attack_config.training.device = 'cuda'


    if attack_config.training.wandb.track:
        wandb_run = wandb_helpers.wandb_init(attack_config)
    else:
        wandb_run = None

    target_config, target_weights_path = load_trained_models.get_target_config_and_weights(attack_config)
        
    # Load data
    transform = get_transforms(target_config, train=False)
    
    train_dataset, _, _ = get_datasets(config=target_config,
                                                            train_transform=transform,
                                                            test_transform=None)
    
    train_loader, val_loader, test_loader = get_data_loaders(target_config)
    
    target_model = get_model(target_config)

    # Load defense
    target_model = get_defense(config=target_config, model=target_model)

    # Load model weights
    target_model.load_model(target_weights_path)

    test_loss, test_accuracy = target_model.evaluate(test_loader)
    print("test_loss", test_loss)
    print("test_accuracy", test_accuracy)

    print()
    
    if attack_config.training.wandb.track:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_accuracy": test_accuracy})

    if attack_config.model.name == "plug_and_play":
        
        evaluation_config, evaluation_weights_path = load_trained_models.get_evaluation_config_and_weights(attack_config)
        
        # Load evaluation model
        evaluation_model = get_model(evaluation_config)
        evaluation_model.load_model(evaluation_weights_path)
        
        # Convert to plug and play compatibility 
        target_model = model_compatibility_wrapper(model = target_model, target_config = target_config)
        evaluation_model = model_compatibility_wrapper(model = evaluation_model, target_config = evaluation_config)
        
        new_attack_config_path = convert_configs(target_config, attack_config)
        new_attack_config = AttackConfigParser(new_attack_config_path)
        
        attack(
            config = new_attack_config,
            target_dataset = train_dataset,
            target_model = target_model,
            evaluation_model = evaluation_model,
            target_config = target_config,
            wandb_run = wandb_run,
            )
   
    print("done")

    
if __name__ == "__main__":
    run_model_inversion()
