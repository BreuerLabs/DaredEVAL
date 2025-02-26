from classifiers.cnn import CNN
from classifiers.mlp import MLP
from classifiers.pretrained import PreTrainedClassifier

def get_model(config, name=None):

    model_name = name if name is not None else config.model.name # option to pass in a model name that overrides config

    if model_name == "CNN":
        model = CNN(config)
        
    elif model_name == "MLP":
        model = MLP(config)
    
    elif model_name == "PreTrainedClassifier":
        model = PreTrainedClassifier(config)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model