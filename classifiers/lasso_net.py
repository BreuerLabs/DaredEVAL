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
        X_train = torch.Tensor([])
        y_train = torch.Tensor([])

        X_val = torch.Tensor([])
        y_val = torch.Tensor([])

        for bidx, (data, target) in enumerate(train_loader):
            X_train = torch.cat((X_train, data))
            y_train = torch.cat((y_train, data))

        for bidx, (data, target) in enumerate(val_loader):
            X_val = torch.cat((X_val, data))
            y_val = torch.cat((y_val, data))

        import pdb; pdb.set_trace()
        print("Best model scored", self.model.score(val_set))

        


    