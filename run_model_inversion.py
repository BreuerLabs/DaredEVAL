from omegaconf import OmegaConf
import hydra
import wandb
import os
import torch
import sys

from classifiers.get_model import get_model
from data_processing.dataloader import get_data_loaders

# import model_inversion.plug_and_play.attack as pnp

from model_inversion.plug_and_play.modify_to_pnp_repo import model_compatability_wrapper


from utils import wandb_helpers

@hydra.main(config_path="configuration/model_inversion", config_name="config.yaml", version_base="1.3")
def run_model_inversion(attack_config):

    if attack_config.training.wandb.track:
        wandb_helpers.wandb_init(attack_config)
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(attack_config)
    
    # If targeting a model trained with wandb tracking
    if attack_config.target_wandb_id:
        target_config, run_name = wandb_helpers.get_target_config(attack_config)

    # If targeting a local run
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
    
    target_model = get_model(target_config)
    
    model_weights_path = f"classifiers/saved_models/{run_name}.pth" #! We can change this to save weights in wandb instead.
    
    target_model.load_model(model_weights_path)
    
    test_loss, test_accuracy = target_model.evaluate(test_loader)
    
    print("test_loss", test_loss)
    print("test_accuracy", test_accuracy)
    
    if target_config.training.wandb.track:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_accuracy": test_accuracy})

    modified_model = model_compatability_wrapper(model = target_model, target_config = target_config)
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