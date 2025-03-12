from classifiers.abstract_classifier import AbstractClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcapt.lca import LCAConv2D

def apply_sparse_defense(config, model:AbstractClassifier):
    
    class SparseDefense(model.__class__):
        
        def __init__(self, config):
            super(SparseDefense, self).__init__(config)
            
            self.sparse_layer = LCAConv2D(in_neurons=config.defense.in_neurons,
                                          out_neurons=config.defense.out_neurons,
                                          kernel_size=config.defense.kernel_size,
                                          stride=config.defense.stride,
                                          pad=config.defense.pad,
                                          lambda_=config.defense.lambda_,
                                          tau=config.defense.tau,
                                          eta=config.defense.eta,
                                          lca_iters=config.defense.lca_iters)
            
            if config.defense.disable_first_module:
                model_children = list(self.feature_extractor.named_children())
                first_layer_name, first_layer = model_children[0]
                setattr(self.feature_extractor, first_layer_name, nn.Identity())
                
                print(f"\n{first_layer_name} disabled!")
            
        def train_model(self, train_loader, val_loader):
            if torch.cuda.device_count() > 1:
                self.sparse_layer = nn.DataParallel(self.sparse_layer)
            return super(SparseDefense, self).train_model(train_loader, val_loader)

        def forward(self, x):
            x = self.sparse_layer(x)
            x = super(SparseDefense, self).forward(x)
            return x
    
    sparse_defended_model = SparseDefense(config)
    
    return sparse_defended_model

