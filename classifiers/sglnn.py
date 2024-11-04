# Smoothed Group Lasso NN (Zhang et al. 2020)

import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torchvision
from torchvision import transforms


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
            # TODO add a 'log sigmoid' activation? 
        )

        return model
    
    def SGL_penalty(self): # Smoothed Group Lasso penalty
        alpha = self.config.model.hyper.sgl_alpha
        w_first = self.first_layer.weight # (n_hidden, input_dim)
        w_norms = torch.linalg.norm(w_first, dim=0) # (1, input_dim)
        smoothed_norms = (w_norms**2 / (2*alpha)) + (alpha/2) # see eqn. (15) in SGLasso paper
        final_smoothed_norms = torch.zeros_like(w_norms)
        final_smoothed_norms[w_norms < alpha] = smoothed_norms[w_norms < alpha]
        final_smoothed_norms[w_norms >= alpha] = w_norms[w_norms >= alpha] # only use smoothed norms if less than alpha
        gl_pen = final_smoothed_norms.sum()
        return gl_pen
    
    
    
    def train_one_epoch(self, train_loader, val_loader, epoch):
        config = self.config
        self.train()
        total_loss = 0
        loss_calculated = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target) + config.model.hyper.gl_lmbda * self.SGL_penalty() # group lasso - TODO add smoothing
            total_loss += loss.item() * len(data)
            loss_calculated += len(data)

            if config.training.verbose == 2:
                print("loss: ", loss.item())

            loss.backward()
            self.optimizer.step()

        train_loss = total_loss / loss_calculated

        return train_loss
    

    def forward(self, x): # (batch_size, channel, height, width)
        bsz, n_channels, height, width = x.shape
        x = torch.reshape(x, (bsz, n_channels*height*width))
        return self.model(x)
