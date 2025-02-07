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
from model_inversion.plug_and_play.our_pnp_utils import *


def pnp_evaluate(evaluation_model,
                w_optimized_unselected,
                final_w,
                final_targets,
                synthesis,
                config,
                target_config,
                targets,
                device,
                idx_to_class,
                batch_size,
                batch_size_single,
                run_id,
                target_dataset,
                rtpt=None):
    # Set devices
    torch.set_num_threads(24)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]
   
    ####################################
    #         Attack Accuracy          #
    ####################################

    # Compute attack accuracy with evaluation model on all generated samples

    # try:
    if not evaluation_model: #! Only relevant if we delete evaluation_model earlier in script to save memory
        #evaluation_model = config.create_evaluation_model() #! TODO: replace
        raise NotImplementedError("not implemented yet!")
    
    
    evaluation_model = torch.nn.DataParallel(evaluation_model)
    evaluation_model.to(device)
    evaluation_model.eval()
    class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                    device=device)

    acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
        w_optimized_unselected,
        targets,
        synthesis,
        config,
        batch_size=batch_size * 2,
        resize=299, #! TODO: Use parameter
        rtpt=rtpt)

    if config.logging:
        try:
            filename_precision = write_precision_list(
                filename = f'model_inversion/plug_and_play/results/precision_list_unfiltered/{run_id}',
                precision_list = precision_list
                )
            
            wandb.save(filename_precision)
        except:
            pass
    print(
        f'\nUnfiltered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}',
        f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
    )

    # Compute attack accuracy on filtered samples
    if config.final_selection:
        acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
            final_w,
            final_targets,
            synthesis,
            config,
            batch_size=batch_size * 2,
            resize=299, #! TODO: Use parameter
            rtpt=rtpt)
        if config.logging:
            filename_precision = write_precision_list(
                f'model_inversion/plug_and_play/results/precision_list_filtered/{run_id}',
                precision_list)
            wandb.save(filename_precision)

        print(
            f'Filtered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}, ',
            f'accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
        )
        # del evaluation_model #! Maybe add again if memory becomes a problem

    # except Exception:
    #     print(traceback.format_exc())


    
    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    fid_score = None
    precision, recall = None, None
    density, coverage = None, None
    try:
        # set transformations
        crop_size = config.attack_center_crop
        
        ### This is replaced by our dataload functions ###
        # target_transform = T.Compose([
        #     T.ToTensor(),
        #     T.Resize((299, 299), antialias=True),
        #     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        
        # training_dataset = create_target_dataset(target_dataset,
        #                                          target_transform)        
        
        target_transform_list = [T.Resize((299, 299), antialias=True)] #! TODO: Use parameters
        
        # This is to only use the transformation shown above
        target_config_no_aug = target_config
        target_config_no_aug.dataset.augment_data = False
        
        target_transform  = get_transforms(target_config_no_aug, target_transform_list, train=False) # Train is false to not get the data augmentations
        training_dataset, _, _ = get_datasets(target_config, train_transform= target_transform, test_transform=None)
        
        # create datasets
        attack_dataset = TensorDataset(final_w, final_targets)
        attack_dataset.targets = final_targets 
        
        training_dataset = ClassSubset(     
            training_dataset,
            target_classes=torch.unique(final_targets).cpu().tolist())

        # compute FID score #! TODO: Use parameters
        fid_evaluation = FID_Score(training_dataset, 
                                   attack_dataset,
                                   device=device,
                                   crop_size=crop_size,
                                   generator=synthesis,
                                   batch_size=batch_size * 3,
                                   dims=2048,
                                   num_workers=8,
                                   gpu_devices=gpu_devices)
        fid_score = fid_evaluation.compute_fid(rtpt)
        print(
            f'FID score computed on {final_w.shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
        )

        # compute precision, recall, density, coverage #! TODO: Use parameters
        prdc = PRCD(training_dataset,
                    attack_dataset,
                    device=device,
                    crop_size=crop_size,
                    generator=synthesis,
                    batch_size=batch_size * 3,
                    dims=2048,
                    num_workers=8,
                    gpu_devices=gpu_devices)
        precision, recall, density, coverage = prdc.compute_metric(
            num_classes=config.num_classes, k=3, rtpt=rtpt)
        print(
            f' Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
        )

    except Exception:
        print(traceback.format_exc())

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_inception = None
    avg_dist_facenet = None
        
    try:
        # Load Inception-v3 evaluation model and remove final layer
        if not evaluation_model:
            evaluation_model_dist = config.create_evaluation_model() #! TODO: Replace this
            
        evaluation_model_dist = evaluation_model
        
        if isinstance(evaluation_model_dist, torch.nn.DataParallel):
            evaluation_model_dist.module.fc = torch.nn.Sequential()  # This is the fix 
        else:
            evaluation_model_dist.model.fc = torch.nn.Sequential() # This doesn't work
            evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist,
                                                      device_ids=gpu_devices)
        
        evaluation_model_dist.to(device)
        evaluation_model_dist.eval()

        ### Get our dataset
        target_transform_list = [T.Resize((299, 299), antialias=True)] #! TODO: Use parameters TODO: Maybe add cropping
        
        # This is to only use the transformation shown above
        target_config_no_aug = target_config
        target_config_no_aug.dataset.augment_data = False
        
        target_transform  = get_transforms(config=target_config_no_aug, 
                                           extra_augmentations=target_transform_list, 
                                           train=False)
        
        training_dataset, _, _ = get_datasets(config=target_config, 
                                              train_transform=target_transform,
                                              test_transform=None)
        ### 
        
        # # Compute average feature distance on Inception-v3
        # evaluate_inception = DistanceEvaluation(evaluation_model_dist,
        #                                         synthesis, 299, #! TODO: use parameter 
        #                                         config.attack_center_crop,
        #                                         target_dataset, config.seed)
        
        # Compute average feature distance on Inception-v3
        evaluate_inception = DistanceEvaluation(evaluation_model_dist,
                                                synthesis, 299, #! TODO: use parameter 
                                                config.attack_center_crop,
                                                training_dataset, config.seed)
        
        avg_dist_inception, mean_distances_list = evaluate_inception.compute_dist(
            final_w,
            final_targets,
            batch_size=batch_size_single * 5,
            rtpt=rtpt)

        if config.logging:
            try:
                filename_distance = write_precision_list(
                    f'model_inversion/plug_and_play/results/distance_inceptionv3_list_filtered/{run_id}',
                    mean_distances_list)
                wandb.save(filename_distance)
                
            except:
                pass

        print('Mean Distance on Inception-v3: ',
              avg_dist_inception.cpu().item())
        
        # Compute feature distance only for facial images #! TODO: Still not modified
        if target_config.dataset.face_dataset:
            # Load FaceNet model for face recognition
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
            facenet.to(device)
            facenet.eval()

            # # Compute average feature distance on facenet #! Replaced
            # evaluater_facenet = DistanceEvaluation(facenet, synthesis, 160,
            #                                        config.attack_center_crop,
            #                                        target_dataset, config.seed)
            
            ### Get our dataset
            face_net_img_size = 160 #! TODO: use parameter
            
            target_transform_list = [T.Resize((face_net_img_size, face_net_img_size), antialias=True)] #! TODO: Maybe add cropping
            
            # This is to only use the transformation shown above
            target_config_no_aug = target_config
            target_config_no_aug.dataset.augment_data = False
            
            target_transform  = get_transforms(config=target_config_no_aug, 
                                               extra_augmentations=target_transform_list, 
                                               train=False)
            
            training_dataset_facenet, _, _ = get_datasets(config=target_config, 
                                                          train_transform=target_transform, 
                                                          test_transform=None)
            ### 
                
            # Compute average feature distance on vggface
            evaluater_facenet = DistanceEvaluation(facenet, synthesis, face_net_img_size,  
                                                    config.attack_center_crop,
                                                    training_dataset_facenet, config.seed)
            
            
            avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
                final_w,
                final_targets,
                batch_size=batch_size_single * 8, #! Hardcoded?
                rtpt=rtpt)
            if config.logging:
                filename_distance = write_precision_list(
                    f'model_inversion/plug_and_play/results/distance_facenet_list_filtered/{run_id}',
                    mean_distances_list)
                wandb.save(filename_distance)

            print('Mean Distance on FaceNet: ', avg_dist_facenet.cpu().item())
    except Exception:
        print(traceback.format_exc())

    ####################################
    #          Finish Logging          #
    ####################################

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

    # Logging of final results
    if config.logging:
        print('Finishing attack, logging results and creating sample images.')
        num_classes = 10 #! Hard coded?
        num_imgs = 8     #! Hard coded?
        # Sample final images from the first and last classes
        label_subset = set(
            list(set(targets.tolist()))[:int(num_classes / 2)] +
            list(set(targets.tolist()))[-int(num_classes / 2):])
        log_imgs = []
        log_targets = []
        log_predictions = []
        log_max_confidences = []
        log_target_confidences = []
        # Log images with smallest feature distance
        for label in label_subset:
            mask = torch.where(final_targets == label, True, False)
            w_masked = final_w[mask][:num_imgs]
            imgs = create_image(w_masked,
                                synthesis,
                                crop_size=config.attack_center_crop,
                                resize=config.attack_resize)
            log_imgs.append(imgs)
            log_targets += [label for i in range(num_imgs)]
            log_predictions.append(torch.tensor(predictions)[mask][:num_imgs])
            log_max_confidences.append(
                torch.tensor(maximum_confidences)[mask][:num_imgs])
            log_target_confidences.append(
                torch.tensor(target_confidences)[mask][:num_imgs])

        log_imgs = torch.cat(log_imgs, dim=0)
        log_predictions = torch.cat(log_predictions, dim=0)
        log_max_confidences = torch.cat(log_max_confidences, dim=0)
        log_target_confidences = torch.cat(log_target_confidences, dim=0)

        log_final_images(log_imgs, log_predictions, log_max_confidences,
                         log_target_confidences, idx_to_class)

        # Find closest training samples to final results
        log_nearest_neighbors(log_imgs,
                              log_targets,
                              evaluation_model_dist,
                              'InceptionV3',
                              target_dataset,
                              img_size=299,
                              seed=config.seed)

        # Use FaceNet only for facial images
        facenet = InceptionResnetV1(pretrained='vggface2')
        facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
        facenet.to(device)
        facenet.eval()
        if target_config.dataset.face_dataset:
            log_nearest_neighbors(log_imgs,
                                  log_targets,
                                  facenet,
                                  'FaceNet',
                                  target_dataset,
                                  img_size=160,
                                  seed=config.seed)

        # Final logging
        final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1,
                            acc_top5, avg_dist_facenet, avg_dist_inception,
                            fid_score, precision, recall, density, coverage,
                            final_targets)

