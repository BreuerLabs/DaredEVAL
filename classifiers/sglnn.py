# Smoothed Group Lasso NN (Zhang et al. 2020)

import torch
import torch.nn as nn
import torch.nn.functional as F
from classifiers.abstract_classifier import AbstractClassifier
# from classifiers.defense_utils import get_feature_norms
import wandb


class SGLNN(AbstractClassifier, nn.Module):

    def __init__(self, config):
        super(SGLNN, self).__init__()
        self.config = config
        self.model = self.init_model(config)

    def init_model(self, config):
        super(SGLNN, self).init_model(config)

        n_channels, height, width = config.dataset.input_size

        self.first_layer = nn.Linear(n_channels*height*width, config.model.hyper.hidden_size) # inputs are flattened along channels and height/width before being passed in
        self.hidden_layer = nn.Linear(config.model.hyper.hidden_size, config.dataset.n_classes)
        
        model = nn.Sequential(
            self.first_layer,
            nn.Tanh(),
            self.hidden_layer,
            nn.LogSigmoid(),
            # TODO is 'log sigmoid' activation right? 
        )

        return model

    def SGL_penalty(self): # Smoothed Group Lasso penalty
        
        alpha = self.config.model.hyper.sgl_alpha
        w_first = self.model[0].weight.data # (n_hidden, input_dim)
        w_norms = torch.linalg.norm(w_first, dim=0) # (1, input_dim)

        if self.config.model.hyper.smooth:
            smoothed_norms = (w_norms**2 / (2*alpha)) + (alpha/2) # see eqn. (15) in SGLasso paper
            final_smoothed_norms = torch.zeros_like(w_norms)
            final_smoothed_norms[w_norms < alpha] = smoothed_norms[w_norms < alpha]
            final_smoothed_norms[w_norms >= alpha] = w_norms[w_norms >= alpha] # only use smoothed norms if less than alpha
            gl_pen = final_smoothed_norms.sum()
        else:
            gl_pen = w_norms.sum()

        return gl_pen
    
    def apply_threshold(self):
        # with torch.no_grad():
        current_w_first = self.model[0].weight.data
        current_norms = torch.linalg.norm(current_w_first, dim=0)
        threshold = self.config.model.hyper.gl_theta * self.config.model.hyper.hidden_size # theta * n_hidden
        below_threshold = current_norms < threshold  # which features we will zero out
        new_w_first = current_w_first * ~below_threshold # (n_hidden, input_dim) x (input_dim) = (n_hidden, input_dim)
        self.model[0].weight.data = new_w_first
    
    def train_one_epoch(self, train_loader):
        config = self.config
        self.train()
        total_loss = 0
        loss_calculated = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            gl_pen = config.model.hyper.gl_lambda * self.SGL_penalty()
            if config.model.criterion == "MSE":
                target = F.one_hot(target, num_classes=config.dataset.n_classes).float() # MSELoss expects one-hot encoded vectors
            loss = self.criterionSum(output, target) + gl_pen
            total_loss += loss.item()
            loss_calculated += len(data)

            if config.training.verbose == 2:
                print("loss: ", loss.item())

            loss.backward()
            self.optimizer.step()
            self.apply_threshold() # zero out weights in first layer that are below a certain threshold

            track_features = self.config.training.wandb.track_features
            if track_features:
                feature_norms = get_feature_norms(self.model, track_features)
                for idx, feature_norm in zip(track_features, feature_norms):
                    wandb.log({f"feature_{idx}" : feature_norm.item()})

        train_loss = total_loss / loss_calculated

        return train_loss
    

    def evaluate(self, loader):
        self.eval()
        
        correct = 0
        total_loss = 0
        total_instances = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)

                loss_target = target if self.config.model.criterion != "MSE" else F.one_hot(target, num_classes=10).float() # MSELoss expects one-hot encoded vectors

                gl_pen = self.config.model.hyper.gl_lambda * self.SGL_penalty()
                loss = self.criterionSum(output, loss_target) + gl_pen
                
                total_loss += loss.item()
                total_instances += len(data)
                
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()
        
        avg_loss = total_loss / total_instances
        accuracy = correct / total_instances
        
        return avg_loss, accuracy

    def forward(self, x): # (batch_size, channel, height, width)
        bsz, n_channels, height, width = x.shape
        x = torch.reshape(x, (bsz, n_channels*height*width))
        return self.model(x)
