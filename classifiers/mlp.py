import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier

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

class MLP(AbstractClassifier, nn.Module):
    
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.model = self.init_model(config)
        
    def init_model(self, config):
        super(MLP, self).init_model(config)
        
        # Input features based on the dataset
        in_features = config.dataset.input_size[1] * config.dataset.input_size[2] * config.dataset.input_size[0]
        
        first_layer = FullyConnectedBlock(in_features, config.model.hyper.n_neurons, config.model.hyper.dropout)
        
        middle_layers = [FullyConnectedBlock(config.model.hyper.n_neurons, config.model.hyper.n_neurons, config.model.hyper.dropout) for _ in range(config.model.hyper.n_depth)]
        
        last_layer = nn.Linear(config.model.hyper.n_neurons, config.dataset.n_classes)
        
        model = nn.Sequential(
                            first_layer,
                            *middle_layers,
                            last_layer
                            )

        return model

    def forward(self, X):
        # Print the shape before flattening for debugging
        print(f"Input shape before flattening: {X.shape}")
        
        # Ensure the input tensor is flattened correctly
        X = X.view(X.size(0), -1)  # Flatten the input tensor
        print(f"Input shape after flattening: {X.shape}")
        return self.model(X)
