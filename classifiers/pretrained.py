# Adapted from PnP classifier.py

import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torchvision
from torchvision import transforms
from torchvision.models import densenet, inception, resnet


class PreTrainedClassifier(AbstractClassifier):
    
    def __init__(self, config):
        super(PreTrainedClassifier, self).__init__()
        self.config = config
        self.model = self.init_model()
        
    def init_model(self):
        super(PreTrainedClassifier, self).init_model(self.config)
        self.n_channels = self.config.dataset.input_size[0]

        arch = self.config.model.architecture
        pretrained = self.config.model.pretrained
        if arch == 'resnet152':
            weights = resnet.ResNet152_Weights.DEFAULT if pretrained else None
            model = resnet.resnet152(weights=weights)
        elif arch == 'resnet18':
            weights = resnet.ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet.resnet18(weights=weights)       
        else:
            raise RuntimeError(
                f'No model with the name {arch} available'
            )

        #! Is this used? It is not defined to do so in the abstract classifier
        if self.config.dataset.n_classes != model.fc.out_features:
            # exchange last layer to match desired numbers of classes
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        return model
    
