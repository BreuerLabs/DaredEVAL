import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
import wandb

from classifiers.abstract_classifier import AbstractClassifier
from Defend_MI.BiDO.train_HSIC import train_HSIC_main
from Defend_MI.BiDO import utils


class BiDO(AbstractClassifier):

    def __init__(self, config):
        super(BiDO, self).__init__()
        self.config = config
        self.model = super(BiDO, self).init_model(config)


    def train_model(self, trainloader, testloader): # adapted from BiDO repo's train_HSIC.py
        if self.config.training.save_as:
            self.save_as = self.config.training.save_as + ".pth"
        elif self.config.training.wandb.track:
            self.save_as = wandb.run.name + ".pth" 
        else:
            raise ValueError("Please provide a name to save the model when not using wandb tracking")
        
        from argparse import ArgumentParser # bido repo uses argparse, we need to translate our config into 'args' and 'loaded_args' objects

        parser = ArgumentParser(description='train with BiDO')
        parser.add_argument('--measure', default='HSIC', help='HSIC | COCO')
        parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
        parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
        parser.add_argument('--root_path', default='./', help='')
        parser.add_argument('--config_dir', default='./config', help='')
        parser.add_argument('--model_dir', default='./target_model', help='')
        parser.add_argument('--hotstart_model', default=False, help='')
        parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')

        inpt = self.config.model.args
        parser_input = [f"--{k}={v}" for k, v in inpt.items()]
        dataset_name = self.config.dataset.dataset
        parser_input.append(f"--dataset={dataset_name}")
        args = parser.parse_args(args=parser_input)

        loaded_args = OmegaConf.to_container(self.config.model.loaded_args)
        hyper = self.config.model.hyper
        model_name = loaded_args['dataset']['model_name']
        if model_name not in loaded_args.keys():
            loaded_args[model_name] = {}
        for key, value in hyper.items():
            loaded_args[model_name][key] = value

        n_classes = self.config.dataset.n_classes
        loaded_args['dataset']['n_classes'] = n_classes

        model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
        os.makedirs(model_path, exist_ok=True)
        
        self.model = train_HSIC_main(args, loaded_args, trainloader, testloader)
        if self.config.model.criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss().cuda() # useful to have this defined in the class
            self.criterionSum = nn.CrossEntropyLoss(reduction='sum')
        else:
            assert 1 == 0, f"{self.config.model.criterion} not yet supported"

        # TODO save model
    
    def forward(self, x):
        hiddens, out = self.model(x)
        return out
    


        


