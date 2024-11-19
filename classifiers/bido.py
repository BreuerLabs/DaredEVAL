import sys, os
import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
from Defend_MI.BiDO.train_HSIC import train_HSIC_main
from Defend_MI.BiDO import utils


class BiDO(AbstractClassifier):

    def __init__(self, config):
        super(BiDO, self).__init__()
        self.config = config
        self.model = None # call model when it trains

    def train_model(self, trainloader, testloader):
        dataset = self.config.dataset.dataset
        measure = self.config.model.measure
        ktype = self.config.model.ktype
        hsic_training = self.config.model.hsic_training
        root_path = self.config.model.root_path
        config_dir = self.config.model.config_dir
        model_dir = self.config.model.model_dir
        
        from argparse import ArgumentParser

        parser = ArgumentParser(description='train with BiDO')
        inpt = self.config.args
        parser_input = [f"--{k}={v}" for k, v in inpt.items()]
        args = parser.parse_args(parser_input)

        loaded_args = self.config.loaded_args
        hyper = self.config.hyper
        model_name = loaded_args['dataset']['model_name']
        for key, value in hyper.items():
            loaded_args[model_name][key] = value

        n_classes = self.config.dataset.output_size
        loaded_args[model_name]['n_classes'] = n_classes

        train_file = loaded_args['dataset']['train_file']
        test_file = loaded_args['dataset']['test_file']

        model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
        os.makedirs(model_path, exist_ok=True)
        
        trainloader = utils.init_dataloader(loaded_args, train_file, mode='train')
        testloader = utils.init_dataloader(loaded_args, test_file, mode='test')
        
        train_HSIC_main(args, loaded_args, trainloader, testloader)

        # Load model
        self.model = None 
