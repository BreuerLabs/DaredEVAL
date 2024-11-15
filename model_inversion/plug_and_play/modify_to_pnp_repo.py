import torch

from model_inversion.plug_and_play.pnp_repo.models import base_model
from classifiers.abstract_classifier import AbstractClassifier


def convert_train_config(config):
    pass

def convert_attack_config(config):
    pass


class convert_classifier(base_model):
    def __init__(self, model, target_config):
        super().__init__()
        self.device = torch.device(target_config.training.device)
        self.model = model
        self.to(self.device)

    def forward(self, x):
        return self.model.forward(x)

    def fit(self, train_loader, val_loader):
        return self.model.train_model(train_loader, val_loader)

    def evaluate(self, data_loader):
        avg_loss, accuracy = self.model.evaluate(data_loader)
        return accuracy, avg_loss

    def predict(self, x):
        return self.model.predict(x)
    
    





