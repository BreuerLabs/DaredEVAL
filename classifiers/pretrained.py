import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torchvision
from torchvision import transforms
from torchvision.models import densenet, inception, resnet

class PreTrainedClassifier(AbstractClassifier):
    
    def __init__(self, config):
        super(PreTrainedClassifier, self).__init__(config)
        self.config = config
        self.model = self.init_model()
        
    def init_model(self):
        super(PreTrainedClassifier, self).init_model()
        self.n_channels = self.config.dataset.input_size[0]

        arch = self.config.model.architecture.lower()
        pretrained = self.config.model.pretrained
        if 'resnet' in arch:
            if arch == 'resnet152':
                weights = resnet.ResNet152_Weights.DEFAULT if pretrained else None
                model = resnet.resnet152(weights=weights)
                self.zdim = 2048 # output dimension of self.feature_extractor below
            elif arch == 'resnet18':
                weights = resnet.ResNet18_Weights.DEFAULT if pretrained else None
                model = resnet.resnet18(weights=weights)
                self.zdim = 512

            if self.config.dataset.n_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.config.dataset.n_classes)


        elif 'inception' in arch:
            weights = inception.Inception_V3_Weights.DEFAULT if pretrained else None
            model = inception.inception_v3(weights=weights,
                                           aux_logits=True,
                                           init_weights=False)   
            
        else:
            raise RuntimeError(
                f'No model with the name {arch} available'
            )

        return model
    

    
    # Only used when training inception models
    def get_inception_loss(self, inception_output, target):
        output, aux_logits = inception_output
            
        aux_loss = torch.tensor(0.0, device=self.device)
    
        aux_loss += self.criterionSum(aux_logits, target) #! does this need to be criterionSum instead of criterion too?
    
        loss = self.criterionSum(output, target) + aux_loss
        
        return loss
    
    def get_loss(self, output, target): # overwritten from AbstractClassifier

        if isinstance(output, inception.InceptionOutputs):
            loss = self.get_inception_loss(output, target)

        else:
            loss = self.criterionSum(output, target)

        return loss
