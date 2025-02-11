import torch
import wandb
import json
import os

from classifiers.get_model import get_model
from data_processing.data_augmentation import get_transforms
from data_processing.datasets import get_datasets
from model_inversion.plug_and_play.pnp_evaluate import pnp_evaluate
from utils.wandb_helpers import get_config
from utils.load_trained_models import get_target_config_and_weights, get_evaluation_config_and_weights
from Plug_and_Play_Attacks.utils.stylegan import load_generator
from model_inversion.plug_and_play.modify_to_pnp_repo import model_compatibility_wrapper, convert_configs
from Plug_and_Play_Attacks.utils.attack_config_parser import AttackConfigParser
from Plug_and_Play_Attacks.utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                            get_stanford_dogs_idx_to_class)

def pnp_evaluate_from_wandb(entity, project, run_id, evaluation_model,
                            new_run_id=None, new_run_name=None, manual_load=None):

    # Set devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]
    
    attack_config, run_name = get_config(entity, project, run_id)
    target_config, target_weights_path = get_target_config_and_weights(attack_config)
    evaluation_config, evaluation_weights_path = get_evaluation_config_and_weights(attack_config)
    
    # Load evaluation model
    evaluation_model = get_model(evaluation_config)
    evaluation_model.load_model(evaluation_weights_path)
    evaluation_model = model_compatibility_wrapper(model = evaluation_model, target_config = evaluation_config)

    new_attack_config_path = convert_configs(target_config, attack_config)
    new_attack_config = AttackConfigParser(new_attack_config_path)


    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # load final_targets
    if manual_load is not None:
        final_targets = manual_load["final_targets"]
        targets = manual_load["targets"]

        num_candidates = attack_config['model']['candidates']['num_candidates']
        samples_per_target = attack_config['model']['final_selection']['samples_per_target']
        targets = torch.repeat_interleave(torch.Tensor(targets), num_candidates).to(torch.int64)
        final_targets = torch.repeat_interleave(torch.Tensor(final_targets), samples_per_target).to(torch.int64)
    else:
        run_summary_path = run.file("wandb-summary.json").download(replace=True)
        print(f"Downloaded run summary from wandb: {run_summary_path.name}")
        # load run summary
        with open(run_summary_path.name, 'r') as f:
            run_summary = json.load(f)
            final_targets = run_summary["final_targets"]
            final_targets = torch.Tensor(final_targets).to(torch.int64)
            targets = run_summary["targets"]
            targets = torch.Tensor(targets).to(torch.int64)
    print(f"final_targets: {final_targets}")
    print(f"targets: {targets}")

    # load w_optimized_unselected
    w_optimized_unselected_path = run.file(f"model_inversion/plug_and_play/results/optimized_w/{run_id}.pt").download(replace=True)
    print(f"Downloaded optimized_unselected w from wandb: {w_optimized_unselected_path.name}")
    with open(w_optimized_unselected_path.name, 'rb') as f:
        w_optimized_unselected = torch.load(f)

    # load final_w
    final_w_path = run.file(f"model_inversion/plug_and_play/results/optimized_w_selected/{run_id}.pt").download(replace=True)
    print(f"Downloaded final w from wandb: {final_w_path.name}")
    with open(final_w_path.name, 'rb') as f:
        final_w = torch.load(f)

    # Load data
    transform = get_transforms(target_config, train=False)
    target_dataset, _, _ = get_datasets(config=target_config,
                                                            train_transform=transform,
                                                            test_transform=None)

    # Load idx to class mappings
    idx_to_class = None
    if new_attack_config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif new_attack_config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:

        class KeyDict(dict):

            def __missing__(self, key):
                return key

        idx_to_class = KeyDict()

    # load synthesis model
    G = load_generator(attack_config['model']['stylegan_model'], device)
    G.img_channels = target_dataset[0][0].shape[0] # Extracting the number of channels from the dataset
    num_ws = G.num_ws
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = num_ws
    del G

    # batch size stuff
    batch_size_single = new_attack_config.attack['batch_size']
    batch_size = new_attack_config.attack['batch_size'] * max(torch.cuda.device_count(), 1) #! Changed this to max(n, 1) to make it work with cpu.

    try:
        with open("secret.txt", "r") as f:
            os.environ['WANDB_API_KEY'] = f.read().strip()
    except Exception as e:
        print(f"\nCreate a secret.txt file with you wandb API key {e}")
        return    

    new_run = wandb.init(project=project, 
        entity=entity,
        id=new_run_id,
        name=new_run_name,
        resume="never",
    )
        
    print(f"wandb initiated with run id: {new_run.id} and run name: {new_run.name}")
        

    # run evaluation
    eval_results = pnp_evaluate(
        w_optimized_unselected=w_optimized_unselected,
        final_w=final_w,
        final_targets=final_targets,
        evaluation_model=evaluation_model,
        synthesis=synthesis,
        config=new_attack_config,
        target_config=target_config,
        targets=targets,
        device=device,
        gpu_devices=gpu_devices,
        idx_to_class=idx_to_class,
        batch_size=batch_size,
        batch_size_single=batch_size_single,
        target_dataset=target_dataset,
        run_id=new_run.id
    )


if __name__ == "__main__":
    manual_load = {} # else dictionary with final_targets and targets
    manual_load['targets'] = [ 57,  12, 140, 125, 114,  71,  52,  44, 216,   8,   7,  23,  55,  59,
        129, 154,   6, 143,  50, 183, 166, 179, 139, 107,  56, 259, 150, 258,
        207, 222,   1, 194, 206,  40, 178, 108,  87, 407, 344, 375, 437, 317,
        312, 459, 314, 448, 481, 353, 419, 332, 471, 276, 451, 382, 402, 296,
        501, 361, 285, 406, 340, 477, 425, 423, 491, 485, 357, 412, 521, 445,
        282, 514, 434, 323, 462]
    manual_load['final_targets'] = [129, 514, 258, 259, 1, 6, 7, 8, 521, 139, 12, 140, 143, 402,
        276, 150, 407, 23, 406, 154, 282, 412, 285, 419, 166, 423, 40, 296, 425, 44, 434, 50, 179, 52,
        178, 437, 55, 183, 57, 56, 59, 312, 317, 314, 445, 448, 194, 451, 323, 71, 459, 332, 206, 207,
        462, 340, 87, 216, 344, 471, 477, 222, 481, 353, 485, 357, 361, 107, 108, 491, 114, 501, 375, 125, 382]
    # manual_load = None

    assert set(manual_load['targets']) == set(manual_load['final_targets'])
    
    run_id = "5ux9y8bf"
    new_run_id = None
    new_run_name = "NewEval-NoDef-75T"

    pnp_evaluate_from_wandb(entity="BreuerLab", project="plug_and_play", evaluation_model="stylegan2_ffhq",
                            run_id=run_id, new_run_id=new_run_id, new_run_name=new_run_name, manual_load=manual_load)