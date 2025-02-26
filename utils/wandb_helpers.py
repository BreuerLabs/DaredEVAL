
import os
from omegaconf import OmegaConf
import wandb
import yaml

def wandb_init(config):

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

        run_id = config.training.wandb.run_id if config.training.wandb.run_id else None
        run_name = config.training.wandb.run_name if config.training.wandb.run_name else None

        run = wandb.init(project=config.training.wandb.project, 
            config=OmegaConf.to_container(config), 
            entity=config.training.wandb.entity,
            id=run_id,
            name=run_name,
            resume="allow",
        )
        
        print(f"wandb initiated with run id: {run.id} and run name: {run.name}")
        return run
    
    except Exception as e:
        print(f"\nCould not initiate wandb logger\nError: {e}")
        
        
def get_config(entity, project, run_id):
    
    def remove_value_keys(config):
        """Removes the 'value' key from each top-level key in the dictionary if it exists."""
        cleaned_config = {}
        for key, value in config.items():
            # If the current section has a 'value' key, use it; otherwise, keep it as is
            if isinstance(value, dict) and "value" in value:
                cleaned_config[key] = value["value"]
                
            else:
                cleaned_config[key] = value
                
        return cleaned_config
        
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
        
    # Download the YAML config file
    target_config = run.file("config.yaml").download(replace=True)
    print("Wandb config downloaded as 'config.yaml':")
    
    with open(target_config.name, 'r') as f:
        target_config_data = yaml.safe_load(f)
        print("Config contents:", target_config_data)
    
    target_config = remove_value_keys(target_config_data)
    target_config = OmegaConf.create(target_config)
    
    return target_config, run.name
    
    
def get_weights(entity, project, run_id, save_as = None):

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    filename = save_as if save_as else run.name
    file_path = f"classifiers/saved_models/{filename}.pth"
    
    target_weights = run.file(file_path).download(replace=True)
    del target_weights

    return file_path
