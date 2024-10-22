import torch

class AbstractClassifier:
    def __init__(self):
        raise NotImplementedError ("this is an abstract class and should not be directly instantiated.")


    def train(self, args):
        pass


    def forward(self, X):
        pass

    
    def predict(self, X):
        return 