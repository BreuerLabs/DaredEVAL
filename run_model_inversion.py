from omegaconf import OmegaConf
import hydra
import wandb
import os
import torch

from classifiers.get_model import get_model
from dataloaders.get_data_loaders import get_data_loaders
from model_inversion.plug_and_play.gan import GAN
from model_inversion.plug_and_play.stylegan import load_discrimator, load_generator
from utils import wandb_helpers

@hydra.main(config_path="configuration/model_inversion", config_name="config.yaml", version_base="1.3")
def run_model_inversion(config):

    if config.training.wandb.track:
        wandb_helpers.wandb_init(config)
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # If targeting a model trained with wandb tracking
    if config.target_wandb_id:
        target_config, run_name = wandb_helpers.get_target_config(config)

    # If targeting a local run
    else:
        return #! TODO: Implement for local runs
    
    target_model = get_model(target_config)
    
    model_weights_path = f"classifiers/saved_models/{run_name}.pth" #! We can change this to save weights in wandb instead.
    
    target_model.load_model(model_weights_path)
    
    test_loss, test_accuracy = target_model.evaluate(test_loader)
    
    if target_config.training.wandb.track:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_accuracy": test_accuracy})
        
        
    


    # # load the model
    # gan = GAN(config)
    
    # # Train model
    # gan.train(train_loader)
    
    # # Load pre-trained StyleGan2 components
    # G = load_generator(config.stylegan_path)
    # D = load_discrimator(config.stylegan_path)
    # num_ws = G.num_ws
    
    # target_model
    
    
    
    print("done")
    
    
if __name__ == "__main__":
    run_model_inversion()