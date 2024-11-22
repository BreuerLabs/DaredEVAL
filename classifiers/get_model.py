
from classifiers.cnn import CNN
from classifiers.mlp import MLP

def get_model(config):
    if config.model.name == "CNN":
        model = CNN(config)
        
    elif config.model.name == "MLP":
        model = MLP(config)
        
    elif config.model.name == "BitNet":
        # from classifiers.bitnet import BitNet 
        # model = BitNet(config)
        raise ValueError(f"BitNet not imported")

    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    
    return model