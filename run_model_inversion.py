from omegaconf import OmegaConf
import hydra
import wandb
import os
import torch
import sys

from classifiers.get_model import get_model
from data_processing.data_loaders import get_data_loaders
from data_processing.datasets import get_datasets

# import model_inversion.plug_and_play.attack as pnp
from Plug_and_Play_Attacks.utils.attack_config_parser import AttackConfigParser
from model_inversion.plug_and_play.our_attack import attack

from model_inversion.plug_and_play.modify_to_pnp_repo import model_compatibility_wrapper, convert_configs


from utils import wandb_helpers

@hydra.main(config_path="configuration/model_inversion", config_name="config.yaml", version_base="1.3")
def run_model_inversion(attack_config):

    if attack_config.training.wandb.track:
        wandb_helpers.wandb_init(attack_config)

    
    # If targeting a model trained with wandb tracking
    if attack_config.target_wandb_id:
        target_config, run_name = wandb_helpers.get_target_config(attack_config)

    # If targeting a local runs
    elif attack_config.target_config_path:
        try:
            target_config = OmegaConf.load(attack_config.target_config_path)
            run_name = target_config.training.save_as
            print("Configuration loaded successfully.")
            
        except Exception as e:
            print(f"Error loading YAML file: {e}", file=sys.stderr)
            raise
        
    else:
        raise ValueError("Please provide either a wandb id or a path to a configuration file")
    
    # Load data
    train_dataset, val_dataset, test_dataset = get_datasets(target_config)
    train_loader, val_loader, test_loader = get_data_loaders(target_config)
    
    target_model = get_model(target_config)
    
    model_weights_path = f"classifiers/saved_models/{run_name}.pth" #! We can change this to save weights in wandb instead.
    
    target_model.load_model(model_weights_path)
    
    test_loss, test_accuracy = target_model.evaluate(test_loader)
    
    print("test_loss", test_loss)
    print("test_accuracy", test_accuracy)

    print()

    
    if attack_config.training.wandb.track:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_accuracy": test_accuracy})

    if attack_config.model.name == "plug_and_play":
        target_model = model_compatibility_wrapper(model = target_model, target_config = target_config)
        new_attack_config_path = convert_configs(target_config, attack_config)
        new_attack_config = AttackConfigParser(new_attack_config_path)
        
        attack(
            config = new_attack_config,
            target_dataset = train_dataset,
            target_model = target_model,
            evaluation_model = None
            )
    
    # if attack_config.model.name == "plug_and_play":
    #     pnp.run(target_model, target_config, attack_config)

    # else:
    #     raise ValueError(f"Unknown model: {attack_config.model.name}")
    
    
    
    # # load the model
    # gan = GAN(config)
    
    # # Train model
    # gan.train(train_loader)
    
    
    print("done")
    
    
if __name__ == "__main__":
    run_model_inversion()