import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torch.nn.functional as F
# from lassonet import LassoNetClassifier
from .prox import inplace_prox, prox


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(FullyConnectedBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, x):
        return self.block(x)


class LassoNet(AbstractClassifier): # adapted from LassoNet Github model.py

    def __init__(self, config):
        self.config = config
        super(LassoNet, self).__init__()

        self.model, self.skip = self.init_model(config) # TODO change structure?
        

    def init_model(self, config):
        super(LassoNet, self).init_model(self.config)
        dropout = config.model.hyper.dropout
        in_features = config.dataset.input_size[1] * config.dataset.input_size[2] * config.dataset.input_size[0]
        n_neurons = config.model.hyper.n_neurons
        n_depth = config.model.hyper.n_depth
        first_layer = FullyConnectedBlock(in_features, n_neurons, dropout)

        middle_layers = [FullyConnectedBlock(n_neurons, n_neurons, dropout) for _ in range(n_depth)]
        
        last_layer = nn.Linear(n_neurons, config.dataset.n_classes)


        model = nn.Sequential(
            first_layer,
            *middle_layers,
            last_layer
        )

        skip = nn.Linear(in_features, config.dataset.n_classes, bias=False)

        return model, skip
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1))

        return self.model(x) + self.skip(x)
    
    def prox(self, *, lambda_, lambda_bar=0, M=1): # not sure what this does
        with torch.no_grad():
            inplace_prox(
                beta=self.skip,
                theta=self.model[0].block[0],
                lambda_=lambda_,
                lambda_bar=lambda_bar,
                M=M,
            )
    
    def skip_GL_penalty(self): # Group Lasso penalty on skip connection [See LassoNet Github page]
        
        w_skip = self.skip.weight.data # (n_hidden, input_dim)
        w_skip_norms = torch.linalg.norm(w_skip, dim=0) # (1, input_dim)
        skip_gl_pen = w_skip_norms.sum()
        return skip_gl_pen
    
    def train_one_epoch(self, train_loader):
        config = self.config
        self.train()
        total_loss = 0
        loss_calculated = 0
        lambda_ = config.model.hyper.skip_gl_lambda

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            skip_gl_pen = lambda_ * self.skip_GL_penalty()
            if config.model.criterion == "MSE":
                target = F.one_hot(target, num_classes=config.dataset.n_classes).float() # MSELoss expects one-hot encoded vectors
            loss = self.criterionSum(output, target) + skip_gl_pen
            total_loss += loss.item()
            loss_calculated += len(data)

            if config.training.verbose == 2:
                print("loss: ", loss.item())

            loss.backward()
            self.optimizer.step()
            # Proximal stuff
            self.prox(
                    lambda_=lambda_ * self.optimizer.param_groups[0]["lr"],
                    M=config.model.hyper.M,
                )

        train_loss = total_loss / loss_calculated
        skip_norms = torch.linalg.norm(self.skip.weight.data, dim=0)
        first_layer_norms = torch.linalg.norm(self.model[0].block[0].weight.data, dim=0)
        print("Features remaining skip: ", len(skip_norms) - (skip_norms < 1e-7).sum())
        print("Features remaining first: ", len(first_layer_norms) - (first_layer_norms < 1e-7).sum())

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

                loss_target = target if self.config.model.criterion != "MSE" else F.one_hot(target, num_classes=self.config.dataset.n_classes).float() # MSELoss expects one-hot encoded vectors

                skip_gl_pen = self.config.model.hyper.skip_gl_lambda * self.skip_GL_penalty()
                loss = self.criterionSum(output, loss_target) + skip_gl_pen
                
                total_loss += loss.item()
                total_instances += len(data)
                
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()
        
        avg_loss = total_loss / total_instances
        accuracy = correct / total_instances
        
        return avg_loss, accuracy

        


    