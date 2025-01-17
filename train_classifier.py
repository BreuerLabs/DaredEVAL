
import hydra
import wandb
import os
import torch
import time
from omegaconf import OmegaConf
import torch.nn as nn

from data_processing.data_loaders import get_data_loaders
from classifiers.get_model import get_model
from defenses.get_defense import get_defense
from utils import wandb_helpers
from utils.lambdalabs.terminate import terminate_lambdalabs_instance

@hydra.main(config_path="configuration/classifier", config_name="config.yaml", version_base="1.3")
def train_classifier(config):
    
    if torch.cuda.is_available() and config.training.device != "cuda":
        question = f"\nCuda is available but not configured from command to be used! Do you wish to use cuda instead of {config.training.device}?\nType y to use cuda, enter if not:"
        
        use_cuda = input(question)
        
        if use_cuda.lower().strip() == "y":
            config.training.device = 'cuda'
            

    if config.training.wandb.track:
        wandb_helpers.wandb_init(config)
    
    # Load the data
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Load the model
    model = get_model(config)
    
    # Load defense
    model = get_defense(config=config, model=model)
    
    # Load trained model weights if given
    if config.training.wandb.load_from_run_id:
        model_weights_path = wandb_helpers.get_weights(
                                                        entity   = config.training.wandb.entity,
                                                        project  = config.training.wandb.project,
                                                        run_id = config.training.wandb.load_from_run_id,
        )
        model.load_model(model_weights_path)

    # Train the model
    model.train_model(train_loader, val_loader)
        
    test_loss, test_accuracy = model.evaluate(test_loader)
    
    if config.training.wandb.track:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_accuracy": test_accuracy})
        
    # Save the configuration for later use   
    OmegaConf.save(config, f"classifiers/saved_configs/{model.save_as}.yaml")

    # Save weights in wandb
    if config.training.wandb.track:
        wandb.save(f"classifiers/saved_models/{model.save_as}")

    # Terminate LL instance if applicable
    if config.LL_terminate_on_end:
        if config.LL_sleep_before_terminate:
            print("Sleeping before LL termination... ")
            time.sleep(int(config.LL_sleep_before_terminate))
        print("Terminating current Lambda Labs instance... ")
        terminate_lambdalabs_instance()
    
if __name__ == "__main__":
    train_classifier()