
from classifiers.cnn import CNN
from classifiers.mlp import MLP
from classifiers.bitnet import BitNet 
from classifiers.lassonet_mlp import LassoNetMLP
from classifiers.sglnn import SGLNN

def get_model(config):
    if config.model.name == "CNN":
        model = CNN(config)
        
    elif config.model.name == "MLP":
        model = MLP(config)
        
    elif config.model.name == "BitNet":
        model = BitNet(config)
    
    elif config.model.name == "LassoNetMLP":
        model = LassoNetMLP(config)

    elif config.model.name == "SGLNN":
        model = SGLNN(config)

    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    
    return model