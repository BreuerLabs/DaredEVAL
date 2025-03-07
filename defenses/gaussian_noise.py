from classifiers.abstract_classifier import AbstractClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

def apply_gaussian_noise_defense(config, model:AbstractClassifier):
    
    class GaussianNoise(model.__class__):
        
        def __init__(self, config):
            super(GaussianNoise, self).__init__(config)

            self.fc_layer = self.model.fc
            self.noise_layer = NoiseLayer(shape=self.model.fc.in_features, stddev=0, seed=self.config.training.seed) # no noise to start
            self.model.fc = nn.Identity() # removes final classification layer

        def forward(self, x): # adapted from DefendMI/BiDO/model.py
            z = self.model(x)
            z_noised = self.noise_layer(z)
            logits = self.fc_layer(z_noised)
            return logits

        def train_model(self, train_loader, val_loader):
            if torch.cuda.device_count() > 1:
                self.fc_layer = nn.DataParallel(self.fc_layer)
                self.noise_layer = nn.DataParallel(self.noise_layer)
            return super(GaussianNoise, self).train_model(train_loader, val_loader)

        def save_model(self, name):
            path = f"classifiers/saved_models/{name}"

            current_noise_layer = self.get_noise_layer()
            if torch.std(current_noise_layer.noise) == 0:
                # add noise to saved model
                new_noise_layer = NoiseLayer(shape=current_noise_layer.noise.shape,
                                            stddev=self.config.defense.stddev,
                                            seed=self.config.training.seed)
            else:
                new_noise_layer = current_noise_layer

            if isinstance(self.model, nn.DataParallel): # self.model, self.noise_layer, and self.fc_layer are on DataParallel

                state = {
                    "model": self.model.module.state_dict(),
                    "noise_layer": new_noise_layer.state_dict(),
                    "fc_layer": self.fc_layer.module.state_dict(),
                }
            else:
                state= {
                    "model": self.model.state_dict(),
                    "noise_layer": new_noise_layer.state_dict(),
                    "fc_layer": self.fc_layer.state_dict(),
                }
        
            torch.save(state, path)
            

        def load_model(self, file_path, map_location = None):
            if map_location is None:
                state = torch.load(file_path, weights_only=True)
            else:
                state = torch.load(file_path, map_location=map_location, weights_only=True)  

            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state['model'])
                self.noise_layer.module.load_state_dict(state['noise_layer'])
                self.fc_layer.module.load_state_dict(state['fc_layer'])
            else:
                self.model.load_state_dict(state['model'])
                self.noise_layer.load_state_dict(state['noise_layer'])
                self.fc_layer.load_state_dict(state['fc_layer'])

        def get_noise_layer(self):
            if isinstance(self.noise_layer, nn.DataParallel):
                return self.noise_layer.module
            else:
                return self.noise_layer

    gaussian_noise_defended_model = GaussianNoise(config)

    return gaussian_noise_defended_model


class NoiseLayer(nn.Module):
    def __init__(self, shape, stddev, seed=42):
        super(NoiseLayer, self).__init__()

        torch.manual_seed(seed)
        self.register_buffer("noise", torch.randn(shape) * stddev)

    def forward(self, x):
        return x + self.noise


