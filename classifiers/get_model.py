
from classifiers.cnn import CNN
from classifiers.mlp import MLP
from classifiers.lassonet_mlp import LassoNetMLP
from classifiers.sglnn import SGLNN
# from classifiers.bitnet import BitNet
from classifiers.bido import BiDO
from classifiers.pretrained import PreTrainedClassifier


def get_model(config):
    if config.model.name == "CNN":
        model = CNN(config)
        
    elif config.model.name == "MLP":
        model = MLP(config)
    
    elif config.model.name == "LassoNetMLP":
        model = LassoNetMLP(config)

    elif config.model.name == "SGLNN":
        model = SGLNN(config)
        
    # elif config.model.name == "BitNet":
    #     model = BitNet(config)

    elif config.model.name == "BiDO":
        model = BiDO(config)
        
    elif config.model.name == "PreTrainedClassifier":
        model = PreTrainedClassifier(config)

    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    
    return model