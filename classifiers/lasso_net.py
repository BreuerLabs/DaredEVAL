import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torchvision
from torchvision import transforms
from lassonet import LassoNetClassifier


class LassoNet(AbstractClassifier):

    def __init__(self, config):
        super(LassoNet, self).__init__()
        self.config = config
        self.model = self.init_model(config)

    def init_model(self, config):
        super(LassoNet, self).init_model(config)
        model = LassoNetClassifier() # TODO add params to config

        return model
    
    def train_model(self, train_loader, val_loader):
        train_set = train_loader.dataset
        val_set = val_loader.dataset

        # import pdb; pdb.set_trace()
        print("Best model scored", self.model.score(val_set))

        


    