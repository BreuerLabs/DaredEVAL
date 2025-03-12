from argparse import ArgumentParser
import torch
import torch.nn as nn

from classifiers.abstract_classifier import AbstractClassifier
from Defend_MI.BiDO import engine

def apply_bido_defense(config, model:AbstractClassifier):

    class BiDOClassifier(model.__class__):

        def __init__(self, config):
            super(BiDOClassifier, self).__init__(config)

            # # remove final classification layer
            # if hasattr(self.model, "fc"):
            #     self.fc_layer = self.model.fc
            #     self.model.fc = nn.Identity()
            # elif hasattr(self.model, "classifier"):
            #     self.fc_layer = self.model.classifier
            #     self.model.classifier = nn.Identity()
            # elif config.model.name == "CNN" or config.model.name == "MLP":
            #     print("Warning: Defense has not been tested well with the 'CNN' and 'MLP' models")
            #     self.fc_layer = list(self.model.children())[-1]
            #     self.model = nn.Sequential(*list(self.model.children())[:-1])
            # else:  
            #     raise ValueError("Cannot recognize the classification layer of the model. Make sure the last layer is named 'fc' or 'classifier'")

            if not config.dataset.val_drop_last: # drop_last must be True in the validation set dataloader because of how BiDO's test_HSIC function is implemented, this is possible to fix but hasn't been done yet
                raise ValueError("dataset.val_drop_last=False is not currently supported for BiDO defense. Please set 'dataset.val_drop_last=True' on the command line.")

        # def train_model(self, train_loader, val_loader):
        #     if torch.cuda.device_count() > 1:
        #         # self.fc_layer = nn.DataParallel(self.fc_layer) # self.model will be put on DataParallel in super train_model call, so we just need to put the fc_layer on DataParallel here

        #     return super(BiDOClassifier, self).train_model(train_loader, val_loader)

        def train_one_epoch(self, train_loader): # adapted from DefendMI/BiDO/train_HSIC.py
            train_loss, train_perc_accuracy = engine.train_HSIC(self,
                                                      self.criterion.cuda(),
                                                      self.optimizer,
                                                      train_loader,
                                                      self.config.defense.a1,
                                                      self.config.defense.a2,
                                                      self.config.dataset.n_classes,
                                                      ktype=self.config.defense.ktype,
                                                      hsic_training=self.config.defense.hsic_training,
                                                      )
            return train_loss
        
        def evaluate(self, loader):
            self.to(self.device)
            loss, perc_accuracy = engine.test_HSIC(self,
                                              self.criterion.cuda(),
                                              loader,
                                              self.config.defense.a1,
                                              self.config.defense.a2,
                                              self.config.dataset.n_classes,
                                              ktype=self.config.defense.ktype,
                                              hsic_training=self.config.defense.hsic_training,
                                              )
            
            return loss, perc_accuracy/100 # convert accuracy from percentage to decimal

        def forward(self, x): # adapted from DefendMI/BiDO/model.py
            z = self.feature_extractor(x)
            logits = self.classification_layer(z)
            return z, logits

        def forward_only_logits(self, x): # necessary for compatibility with attacks
            _, logits = self(x)
            return logits

        def save_model(self, name):
            path = f"classifiers/saved_models/{name}"
            if isinstance(self.model, nn.DataParallel): # self.model, and self.fc_layer are on DataParallel
                state = {
                    "model": self.model.module.state_dict(),
                    "fc_layer": self.fc_layer.module.state_dict(),
                }
            else:
                state= {
                    "model": self.model.state_dict(),
                    "fc_layer": self.fc_layer.state_dict(),
                }
            torch.save(state, path)

        def load_model(self, file_path, map_location = None):
            if map_location is None:
                state = torch.load(file_path, weights_only=True)
            else:
                state = torch.load(file_path, map_location=map_location, weights_only=True)  

            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state['model'])
                self.fc_layer.module.load_state_dict(state['fc_layer'])
            else:
                self.model.load_state_dict(state['model'])
                self.fc_layer.load_state_dict(state['fc_layer'])
        
    bido_defended_model = BiDOClassifier(config)
    return bido_defended_model
        