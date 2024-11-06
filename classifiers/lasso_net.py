import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torchvision
from torchvision import transforms
from lassonet import LassoNetClassifier


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(FullyConnectedBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, x):
        return self.block(x)


class LassoNet(AbstractClassifier): # adapted from LassoNet Github model.py

    def __init__(self, config):
        self.config = config
        super(LassoNet, self).__init__()

        self.model, self.skip = self.init_model(config)
        

    def init_model(self, config):
        super(LassoNet, self).init_model(self.config)
        dropout = config.model.hyper.dropout
        in_features = config.dataset.input_size[1] * config.dataset.input_size[2] * config.dataset.input_size[0]
        n_neurons = config.model.hyper.n_neurons
        n_depth = config.model.hyper.n_depth
        first_layer = FullyConnectedBlock(in_features, n_neurons, dropout)

        middle_layers = [FullyConnectedBlock(n_neurons, n_neurons, dropout) for _ in range(n_depth)]
        
        last_layer = nn.Linear(n_neurons, config.dataset.n_classes)


        model = nn.Sequential(
            first_layer,
            *middle_layers,
            last_layer
        )

        skip = nn.Linear(in_features, config.dataset.n_classes, bias=False)

        return model, skip
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1))

        return self.model(x) + self.skip(x)
    

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

        print("Best model scored", self.model.score(val_set))

        


    