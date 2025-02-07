import argparse
import csv
import math
import random
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import wandb
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset

from Plug_and_Play_Attacks.attacks.final_selection import perform_final_selection
from Plug_and_Play_Attacks.attacks.optimize import Optimization
from Plug_and_Play_Attacks.datasets.custom_subset import ClassSubset
from Plug_and_Play_Attacks.metrics.classification_acc import ClassificationAccuracy
from Plug_and_Play_Attacks.metrics.fid_score import FID_Score
from Plug_and_Play_Attacks.metrics.prcd import PRCD
from Plug_and_Play_Attacks.utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                            get_stanford_dogs_idx_to_class)
from Plug_and_Play_Attacks.utils.stylegan import create_image, load_discrimator, load_generator
from Plug_and_Play_Attacks.utils.wandb import *

from data_processing.datasets import get_datasets
from data_processing.data_augmentation import get_transforms
from model_inversion.plug_and_play.pnp_evaluate import pnp_evaluate
from model_inversion.plug_and_play.our_pnp_utils import *


def attack(config, target_dataset, target_model, evaluation_model, target_config, wandb_run = None):
    ####################################
    #        Attack Preparation        #
    ####################################

    # Set devices
    torch.set_num_threads(24)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # # Define and parse attack arguments
    # parser = create_parser()
    # config, args = parse_arguments(parser)

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
    idx_to_class = None
    if config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:

        class KeyDict(dict):

            def __missing__(self, key):
                return key

        idx_to_class = KeyDict()

    # Load pre-trained StyleGan2 components
    G = load_generator(config.stylegan_model, device)
    D = load_discrimator(config.stylegan_model, device)
    
    G.img_channels = target_dataset[0][0].shape[0] # Extracting the number of channels from the dataset
    
    num_ws = G.num_ws

    # target_dataset = config.get_target_dataset() #! Passed in as argument instead of loading here
    target_model_name = target_model.name
    target_model_num_classes = target_model.num_classes

    # Distribute models
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)

    # Reinstanciate model variables
    target_model.name = target_model_name
    synthesis.num_ws = num_ws
    target_model.num_classes = target_model_num_classes
    
    
    # Load basic attack parameters
    num_epochs = config.attack['num_epochs']
    batch_size_single = config.attack['batch_size']
    batch_size = config.attack['batch_size'] * max(torch.cuda.device_count(), 1) #! Changed this to max(n, 1) to make it work with cpu.
    
    
    config.model = target_model #! Instead of running their load model, we pass it directly 
    
    targets = config.create_target_vector()

    # Create initial style vectors
    w, w_init, x, V = create_initial_vectors(config, G, target_model, targets,
                                             device)
    del G

    # Initialize wandb logging
    if config.logging:
        optimizer = config.create_optimizer(params=[w])
        # wandb_run = init_wandb_logging(optimizer, target_model_name, config) #! wandb_run is passed as argument instead
        
        run_id = wandb_run.id

    # Print attack configuration
    print(
        f'Start attack against {target_model.name} optimizing w with shape {list(w.shape)} ',
        f'and targets {dict(Counter(targets.cpu().numpy()))}.')
    print(f'\nAttack parameters')
    for key in config.attack:
        print(f'\t{key}: {config.attack[key]}')
    print(
        f'Performing attack on {torch.cuda.device_count()} gpus and an effective batch size of {batch_size} images.'
    )

    # Initialize RTPT
    rtpt = None
    # if args.rtpt: #! TODO: Maybe worth enabling as an option
    #     max_iterations = math.ceil(w.shape[0] / batch_size) \
    #         + int(math.ceil(w.shape[0] / (batch_size * 3))) \
    #         + 2 * int(math.ceil(config.final_selection['samples_per_target'] * len(set(targets.cpu().tolist())) / (batch_size * 3))) \
    #         + 2 * len(set(targets.cpu().tolist()))
    #     rtpt = RTPT(name_initials='LS',
    #                 experiment_name='Model_Inversion',
    #                 max_iterations=max_iterations)
    #     rtpt.start()

    # Log initial vectors
    if config.logging:
        Path("results").mkdir(parents=True, exist_ok=True)
        init_w_path = f"model_inversion/plug_and_play/results/init_w/{run_id}.pt"
        torch.save(w.detach(), init_w_path)
        wandb.save(init_w_path)

    # Create attack transformations
    attack_transformations = config.create_attack_transformations()

    ####################################
    #         Attack Iteration         #
    ####################################
    optimization = Optimization(target_model, synthesis, discriminator,
                                attack_transformations, num_ws, config)

    # Collect results
    w_optimized = []

    # Prepare batches for attack
    for i in range(math.ceil(w.shape[0] / batch_size)):
        w_batch = w[i * batch_size:(i + 1) * batch_size].to(device)
        targets_batch = targets[i * batch_size:(i + 1) * batch_size].to(device)
        print(
            f'\nOptimizing batch {i+1} of {math.ceil(w.shape[0] / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
        )

        # Run attack iteration
        torch.cuda.empty_cache()
        w_batch_optimized = optimization.optimize(w_batch, targets_batch,
                                                  num_epochs).detach().cpu()

        if rtpt:
            num_batches = math.ceil(w.shape[0] / batch_size)
            rtpt.step(subtitle=f'batch {i+1} of {num_batches}')

        # Collect optimized style vectors
        w_optimized.append(w_batch_optimized)

    # Concatenate optimized style vectors
    w_optimized_unselected = torch.cat(w_optimized, dim=0)
    torch.cuda.empty_cache()
    del discriminator

    # Log optimized vectors
    if config.logging:
        optimized_w_path = f"model_inversion/plug_and_play/results/optimized_w/{run_id}.pt"
        torch.save(w_optimized_unselected.detach(), optimized_w_path)
        wandb.save(optimized_w_path)

    ####################################
    #          Filter Results          #
    ####################################

    # Filter results
    if config.final_selection:
        print(
            f'\nSelect final set of max. {config.final_selection["samples_per_target"]} ',
            f'images per target using {config.final_selection["approach"]} approach.'
        )
        final_w, final_targets = perform_final_selection(
            w_optimized_unselected,
            synthesis,
            config,
            targets,
            target_model,
            device=device,
            batch_size=batch_size * 10, #? Why multiplied by 10?
            **config.final_selection,
            rtpt=rtpt)
        print(f'Selected a total of {final_w.shape[0]} final images ',
              f'of target classes {set(final_targets.cpu().tolist())}.')
    else:
        final_targets, final_w = targets, w_optimized_unselected
        
    del target_model

    # Log selected vectors
    if config.logging:
        optimized_w_path_selected = f"model_inversion/plug_and_play/results/optimized_w_selected/{run_id}.pt"
        torch.save(final_w.detach(), optimized_w_path_selected)
        wandb.save(optimized_w_path_selected)
        wandb.config.update({'w_path': optimized_w_path})

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
        batch_size_single=batch_size_single,
        run_id=run_id,
        target_dataset=target_dataset,
    )

# if __name__ == '__main__': #! We run attack function from this script in run_model_inversion.py
#     main()
