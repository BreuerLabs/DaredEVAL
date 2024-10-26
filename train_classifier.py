from omegaconf import OmegaConf
import hydra
import wandb
import os
import torch

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def train_classifier(config):
    if config.wandb.track:
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
            run = wandb.init(project=config.wandb.project, 
                    config=OmegaConf.to_container(config), 
                    entity=config.wandb.entity)
            
            print(f"wandb initiated with run id: {run.id} and run name: {run.name}")
        except Exception as e:
            print(f"\nCould not initiate wandb logger\nError: {e}")
            
    
            

if __name__ == "__main__":
    train_classifier()