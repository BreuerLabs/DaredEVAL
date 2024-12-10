# adapted from https://stackoverflow.com/questions/66343862/how-to-create-a-1-to-1-feed-forward-layer

import torch
import torch.nn as nn

# from classifiers.abstract_classifier import AbstractClassifier
# from classifiers.mlp import MLP
# from classifiers.cnn import CNN

class ElementwiseLinear(nn.Module):
    def __init__(self, input_size: tuple) -> None:
        super(ElementwiseLinear, self).__init__()

        # w is the learnable weight of this layer module
        self.weight = nn.Parameter(torch.ones(input_size), requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # simple elementwise multiplication
        return self.weight * x
    

# def lasso_penalty(model): # Lasso penalty on one-dimensional weights
        
#         w_first = model[0].weight.data
#         lasso_pen = w_first.abs().sum()

#         return lasso_pen
    

# def get_feature_norms(model, feature_idxs):
#     w_first = model[0].weight.data
    
#     if isinstance(feature_idxs[0], list): # like drop_layer, one weight for each feature index, norm is just abs. value
#         w_features = [w_first[:, *feature_idx] for feature_idx in feature_idxs]
#         return [torch.abs(w_feature) for w_feature in w_features]
    
#     elif isinstance(feature_idxs[0], int):
#         if len(w_first.shape) == 1: # just one weight for each feature index, norm is just abs. value
#             w_features = [w_first[feature_idx] for feature_idx in feature_idxs]
#             return [torch.abs(w_feature) for w_feature in w_features]
        
#         elif len(w_first.shape) == 2: # like SGLNN, take L2 norm of all weights associated with each feature index
#             w_features = [w_first[:, feature_idx] for feature_idx in feature_idxs]
#             return [torch.linalg.norm(w_feature) for w_feature in w_features]
    
        
    

# class DropLayer(AbstractClassifier, nn.Module):
    
#     def __init__(self, config):
#         super(DropLayer, self).__init__()
#         self.config = config
#         self.model = self.init_model(config)
        
#     def init_model(self, config):
#         super(DropLayer, self).init_model(config)

#         arch = config.model.architecture

#         # Input features based on the dataset
#         if arch == "MLP":     
#             in_features = (config.dataset.input_size[1] * config.dataset.input_size[2] * config.dataset.input_size[0],)
#             drop_layer = ElementwiseLinear(in_features)
#             main_layers = MLP(config)

#         elif arch == "CNN":
#             in_features = (config.dataset.input_size[0], config.dataset.input_size[1], config.dataset.input_size[2])
#             drop_layer = ElementwiseLinear(in_features)
#             main_layers = CNN(config)

#         else:
#             assert 0 == 1, f"architecture not recognized: {arch}"
        
#         model = nn.Sequential(
#                             drop_layer,
#                             main_layers
#                             )
#         return model


#     def forward(self, x):
#         if self.config.model.architecture == "MLP":
#             batch_size = x.shape[0]
#             x = torch.reshape(x, (batch_size, -1))
        
#         else:
#             pass
        
#         return self.model(x)
