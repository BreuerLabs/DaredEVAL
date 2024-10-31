# Smoothed Group Lasso NN (Zhang et al. 2020)

import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torchvision
from torchvision import transforms


class SGLNN(AbstractClassifier, nn.Module):

    def __init__(self, config):
        super(SGLNN, self).__init__()
        self.config = config
        self.model = self.init_model(config)

    def init_model(self, config):
        super(SGLNN, self).init_model(config)

        first_layer = nn.Linear(config.input_size, config.hidden_size)

        hidden_layer = nn.Linear(config.input_size, config.hidden_size)

        return model
    
    def forward(self, x): # (batch_size, channel, height, width)
        bsz, n_channels, height, width = x.shape
        x = torch.reshape(x, (bsz, n_channels*height*width))
        return self.model(x)
