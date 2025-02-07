import torch
import wandb
import json

from classifiers.get_model import get_model
from model_inversion.plug_and_play.pnp_evaluate import pnp_evaluate
from utils.wandb_helpers import get_config
from utils.load_trained_models import get_target_config_and_weights, get_evaluation_config_and_weights
from Plug_and_Play_Attacks.utils.stylegan import load_generator
from model_inversion.plug_and_play.modify_to_pnp_repo import model_compatibility_wrapper, convert_configs
from Plug_and_Play_Attacks.utils.attack_config_parser import AttackConfigParser


def pnp_evaluate_from_wandb(entity, project, run_id, evaluation_model):

    # Set devices
    torch.set_num_threads(24)
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
    run_summary_path = run.file("wandb-summary.json").download(replace=True)
    print(f"Downloaded run summary from wandb: {run_summary_path.name}")
    # load run summary
    with open(run_summary_path.name, 'r') as f:
        run_summary = json.load(f)

    # load w_optimized_unselected
    w_optimized_unselected_path = run.file(f"model_inversion/plug_and_play/results/optimized_w/{run_id}.pt").download(replace=True)
    print(f"Downloaded optimized_unselected w from wandb: {w_optimized_unselected_path.name}")

    with open(w_optimized_unselected_path.name, 'rb') as f:
        w_optimized_unselected = torch.load(f).to(device)

    # load final_w and final_targets
    final_w_path = run.file(f"model_inversion/plug_and_play/results/optimized_w_selected/{run_id}.pt").download(replace=True)
    print(f"Downloaded final w from wandb: {final_w_path.name}")
    final_tar

    # load synthesis model
    G = load_generator(evaluated_run_config.stylegan_model, device)
    G.img_channels = target_dataset[0][0].shape[0] # Extracting the number of channels from the dataset
    num_ws = G.num_ws
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = num_ws
    del G


    eval_results = pnp_evaluate(
        w_optimized_unselected=w_optimized_unselected,
        final_w=final_w,
        final_targets=final_targets,
        evaluation_model=evaluation_model,
        synthesis=synthesis,
        config=config,
        target_config=target_config,
        targets=targets,
        device=device,
        idx_to_class=idx_to_class,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    pnp_evaluate_from_wandb(entity="BreuerLab", project="plug_and_play", run_id="5ux9y8bf", evaluation_model="stylegan2_ffhq")