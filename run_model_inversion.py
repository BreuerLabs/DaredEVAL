from omegaconf import OmegaConf
import hydra
import wandb
import os
import torch
from dataloaders.get_data_loaders import get_data_loaders
from model_inversion.plug_and_play.gan import GAN

@hydra.main(config_path="conf_model_inversion", config_name="config.yaml", version_base="1.3")
def run_model_inversion(config):

    if config.training.wandb.track:
        try:
            with open("secret.txt", "r") as f:
                os.environ['WANDB_API_KEY'] = f.read().strip()
        
        except Exception as e:
            print(f"\nCreate a secret.txt file with you wandb API key {e}")
            return    
        
        print(f"configuration: \n {OmegaConf.to_yaml(config)}")
        # Initiate wandb logger
        try:
            # project is the name of the project in wandb, entity is the username
            # You can also add tags, group etc.
            run = wandb.init(project=config.training.wandb.project, 
                    config=OmegaConf.to_container(config), 
                    entity=config.training.wandb.entity)
            
            print(f"wandb initiated with run id: {run.id} and run name: {run.name}")
        except Exception as e:
            print(f"\nCould not initiate wandb logger\nError: {e}")
            
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # load the model
    gan = GAN(config)
    
    # Train model
    gan.train(train_loader)
    
    
    
    print("done")
    
    
if __name__ == "__main__":
    run_model_inversion()