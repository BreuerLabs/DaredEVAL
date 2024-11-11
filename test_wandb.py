from omegaconf import OmegaConf
import hydra
import wandb
import os
from wandb import Api


@hydra.main(config_path="configuration/model_inversion", config_name="config.yaml", version_base="1.3")
def load_and_print_wandb_config(config):

    # Check if wandb tracking is enabled in config
    if config.training.wandb.track:
        try:
            with open("secret.txt", "r") as f:
                os.environ['WANDB_API_KEY'] = f.read().strip()
        
        except Exception as e:
            print(f"\nCreate a secret.txt file with your wandb API key {e}")
            return

        try:
            # Initialize wandb API and get the run
            api = Api()
            # Replace with your project, entity, and the specific run_id you want to access
            run = api.run(f"{config.training.wandb.entity}/{config.training.wandb.project}/{config.target_wandb_id}")
            
            # Download the YAML config file
            config_yaml = run.file("config.yaml").download(replace=True)
            print("Wandb config downloaded as 'config.yaml':")
            
            with open(config_yaml.name, 'r') as f:
                print(f.read())  # Print the contents of the YAML file
            
        except Exception as e:
            print(f"\nCould not retrieve wandb config\nError: {e}")

if __name__ == "__main__":
    load_and_print_wandb_config()