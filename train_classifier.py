
import hydra
import wandb
import os
import torch
from omegaconf import OmegaConf

from dataloaders.get_data_loaders import get_data_loaders
from classifiers.get_model import get_model
from utils import wandb_helpers

@hydra.main(config_path="configuration/classifier", config_name="config.yaml", version_base="1.3")
def train_classifier(config):

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
    OmegaConf.save(config, f"classifiers/saved_configs/{config.training.save_as}.yaml")
    
    
if __name__ == "__main__":
    train_classifier()