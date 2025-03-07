from argparse import ArgumentParser
import torch
import torch.nn as nn

from classifiers.abstract_classifier import AbstractClassifier
from Defend_MI.BiDO import engine

def apply_bido_defense(config, model:AbstractClassifier):

    class BiDOClassifier(model.__class__):

        def __init__(self, config):
            super(BiDOClassifier, self).__init__(config)

            self.fc_layer = self.model.fc
            self.model.fc = nn.Identity() # removes final classification layer

            if not config.dataset.val_drop_last: # val_drop_last must be True because of how BiDO's test_HSIC function is implemented, this is possible to fix but hasn't been done yet
                raise ValueError("Validation set must have val_drop_last=True for BiDO defense")


        def train_model(self, train_loader, val_loader):
            if torch.cuda.device_count() > 1:
                self.fc_layer = nn.DataParallel(self.fc_layer) # self.model will be put on DataParallel in super train_model call, so we just need to put the fc_layer on DataParallel here

            return super(BiDOClassifier, self).train_model(train_loader, val_loader)


        def train_one_epoch(self, train_loader): # adapted from DefendMI/BiDO/train_HSIC.py
            #! TODO: make sure model is initialized correctly
            #! TODO: does this need a super call? How would that work?
            #! TODO: keep track of self.train_step somehow
            train_loss, train_perc_acc = engine.train_HSIC(self,
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
            self.to(self.device) #! not sure if this line is necessary
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
            z = self.model(x)
            logits = self.fc_layer(z)
            return z, logits

        def forward_only_logits(self, x):
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
        