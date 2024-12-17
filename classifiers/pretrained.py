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

        arch = self.config.model.architecture
        pretrained = self.config.model.pretrained
        if arch.lower() == 'resnet152':
            weights = resnet.ResNet152_Weights.DEFAULT if pretrained else None
            model = resnet.resnet152(weights=weights)
        elif arch.lower() == 'resnet18':
            weights = resnet.ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet.resnet18(weights=weights)       
        
        elif 'inception' in arch.lower():
            weights = inception.Inception_V3_Weights.DEFAULT if pretrained else None
            model = inception.inception_v3(weights=weights,
                                           aux_logits=True,
                                           init_weights=False)
            
            return model    
            
        else:
            raise RuntimeError(
                f'No model with the name {arch} available'
            )

        #! Is this used? It is not defined to do so in the abstract classifier
        if self.config.dataset.n_classes != model.fc.out_features:
            # exchange last layer to match desired numbers of classes
            model.fc = nn.Linear(model.fc.in_features, self.config.dataset.n_classes)

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
        
        if self.config.defense.penalty == "lasso": # normal lasso on first layer weights
            lasso_pen = self.config.defense.lasso.lambda_ * self.lasso_penalty()
            loss = loss + lasso_pen

        return loss
    
    # def train_one_epoch(self, train_loader):
    #     config = self.config
    #     self.train()
    #     total_loss = 0
    #     loss_calculated = 0
        
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         data, target = data.to(self.device), target.to(self.device)
    #         self.optimizer.zero_grad()
            
    #         output = self(data)
            
    #         if isinstance(output, inception.InceptionOutputs):
    #             loss = self.get_inception_loss(output, target)

    #         else:
    #             loss = self.criterion(output, target)
                
    #         total_loss += loss.item() * len(data)
    #         loss_calculated += len(data)

    #         if config.training.verbose == 2:
    #             print("loss: ", loss.item())

    #         loss.backward()
    #         self.optimizer.step()

    #     train_loss = total_loss / loss_calculated

    #     return train_loss
    
            