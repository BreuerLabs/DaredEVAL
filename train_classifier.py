
import hydra
import wandb
import os
import torch
from omegaconf import OmegaConf

from data_processing.data_loaders import get_data_loaders
from classifiers.get_model import get_model
from utils import wandb_helpers

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
    
    model = get_model(config)
    
    model.train_model(train_loader, val_loader)
    
    test_loss, test_accuracy = model.evaluate(test_loader)
    
    if config.training.wandb.track:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_accuracy": test_accuracy})
        
    # Save the configuration for later use   
    OmegaConf.save(config, f"classifiers/saved_configs/{model.save_as}.yaml")

    # Save weights in wandb
    if config.training.wandb.track:
        # wandb.log_model(path=f"classifiers/saved_models/{model.save_as}")
        wandb.save(f"classifiers/saved_models/{model.save_as}")
    
    
if __name__ == "__main__":
    train_classifier()