
# from bitnet import BitLinear

# import torch
# import torch.nn as nn
# from classifiers.abstract_classifier import AbstractClassifier

# class FullyBitConnectedBlock(nn.Module):
#     def __init__(self, in_features, out_features, dropout_rate):
#         super(FullyBitConnectedBlock, self).__init__()
        
#         self.fc = BitLinear(in_features, out_features)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)

#         self.block = nn.Sequential(
#             self.fc,
#             self.relu,
#             self.dropout,
#         )
        
#     def forward(self, x):
#         return self.block(x)

# class BitNet(AbstractClassifier, nn.Module):
    
#     def __init__(self, config):
#         super(BitNet, self).__init__()
#         self.config = config
#         self.model = self.init_model(config)
        
#     def init_model(self, config):
#         super(BitNet, self).init_model(config)
        
#         # Input features based on the dataset
#         in_features = config.dataset.input_size[1] * config.dataset.input_size[2] * config.dataset.input_size[0]
        
#         first_layer = FullyBitConnectedBlock(in_features, config.model.hyper.n_neurons, config.model.hyper.dropout)
        
#         middle_layers = [FullyBitConnectedBlock(config.model.hyper.n_neurons, config.model.hyper.n_neurons, config.model.hyper.dropout) for _ in range(config.model.hyper.n_depth)]
        
#         last_layer = BitLinear(config.model.hyper.n_neurons, config.dataset.n_classes)
        
#         model = nn.Sequential(
#                             first_layer,
#                             *middle_layers,
#                             last_layer
#                             )

#         return model

#     def forward(self, x):
#         # Flatten the input tensor
#         x = x.view(x.size(0), -1)
#         return self.model(x)
