from argparse import ArgumentParser
import torch

from classifiers.abstract_classifier import AbstractClassifier
from Defend_MI.BiDO import engine

def apply_bido_defense(config, model:AbstractClassifier):

    class BiDOClassifier(model.__class__):

        def __init__(self, config):
            super(BiDOClassifier, self).__init__(config)
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        def train_one_epoch(self, train_loader): # adapted from DefendMI/BiDO/train_HSIC.py
            #! TODO: make sure model is initialized correctly
            #! TODO: does this need a super call? How would that work?
            #! TODO: keep track of self.train_step somehow
            train_loss, train_acc = engine.train_HSIC(self.model,
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
            loss, accuracy = engine.test_HSIC(self.model,
                                              self.criterion.cuda(),
                                              loader,
                                              self.config.defense.a1,
                                              self.config.defense.a2,
                                              self.config.dataset.n_classes,
                                              ktype=self.config.defense.ktype,
                                              hsic_training=self.config.defense.hsic_training,
                                              )
            
            return loss, accuracy
        
    bido_defended_model = BiDOClassifier(config)

    return bido_defended_model
        