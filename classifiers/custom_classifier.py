from classifiers.abstract_classifier import AbstractClassifier
import torch.nn as nn

# Implement your custom classifier here, look at cnn.py or ml.py for examples

class MyClassifier(nn.Moduel, AbstractClassifier):
    
    def __init__(self, config):
        super(MyClassifier, self).__init__(config)
        self.config = config
        self.model = self.init_model()

    def init_model(self):
        raise NotImplementedError("Implement the init_model method in your custom classifier")