from classifiers.cnn import CNN
from classifiers.mlp import MLP
from classifiers.lassonet_mlp import LassoNetMLP
from classifiers.sglnn import SGLNN
# from classifiers.bitnet import BitNet
from classifiers.bido import BiDO
from classifiers.pretrained import PreTrainedClassifier
from classifiers.drop_layer import DropLayer

def get_model(config, name=None):

    model_name = name if name is not None else config.model.name # option to pass in a model name that overrides config

    if model_name == "CNN":
        model = CNN(config)
        
    elif model_name == "MLP":
        model = MLP(config)
    
    elif model_name == "LassoNetMLP":
        model = LassoNetMLP(config)

    elif model_name == "SGLNN":
        model = SGLNN(config)
        
    # elif model_name == "BitNet":
    #     model = BitNet(config)

    elif model_name == "BiDO":
        model = BiDO(config)

    elif model_name == "PreTrainedClassifier":
        model = PreTrainedClassifier(config)

    elif model_name == "DropLayer":
        model = DropLayer(config)

    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model