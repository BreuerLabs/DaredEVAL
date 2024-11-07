import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torch.nn.functional as F
from .lasso_net_helper import inplace_prox, prox
import numpy as np
import wandb
from tqdm import tqdm


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

        self.model, self.skip = self.init_model(self.config) # TODO change structure?
        self.lambda_ = self.config.model.hyper.skip_gl_lambda
        self.M = self.config.model.hyper.M
        if self.config.training.wandb.track:
            wandb.log({"skip_strength" : self.M})   

        

    def init_model(self, config):
        super(LassoNet, self).init_model(config)
        dropout = self.config.model.hyper.dropout
        in_features = self.config.dataset.input_size[1] * self.config.dataset.input_size[2] * self.config.dataset.input_size[0]
        n_neurons = self.config.model.hyper.n_neurons
        n_depth = self.config.model.hyper.n_depth
        first_layer = FullyConnectedBlock(in_features, n_neurons, dropout)

        middle_layers = [FullyConnectedBlock(n_neurons, n_neurons, dropout) for _ in range(n_depth)]
        
        last_layer = nn.Linear(n_neurons, self.config.dataset.n_classes)

        model = nn.Sequential(
            first_layer,
            *middle_layers,
            last_layer
        )



        skip = nn.Linear(in_features, self.config.dataset.n_classes, bias=False)

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

    def lambda_start(self):
        """Estimate when the model will start to sparsify."""

        def is_sparse(lambda_):
            with torch.no_grad():
                beta = self.skip.weight.data
                theta = self.model[0].block[0].weight.data

                for _ in range(10000):
                    new_beta, theta = prox(
                        beta,
                        theta,
                        lambda_=lambda_,
                        lambda_bar=self.config.model.hyper.lambda_bar,
                        M=self.config.model.hyper.M,
                    )
                    if torch.abs(beta - new_beta).max() < 1e-5:
                        break
                    beta = new_beta
                return (torch.norm(beta, p=2, dim=0) == 0).sum()

        start = self.config.model.hyper.lambda_start_begin
        factor = self.config.model.hyper.lambda_start_factor
        while not is_sparse(factor * start):
            start *= factor
        return start
    
    def get_feature_norms(self, feature_idxs): # get L2 norm of weights in first layer coming from feature at feature_idx
        w_features = [self.model[0].block[0].weight.data[:, feature_idx] for feature_idx in feature_idxs]
        return [torch.linalg.norm(w_feature) for w_feature in w_features] # L2 norm
    
    def skip_GL_penalty(self): # Group Lasso penalty on skip connection [See LassoNet Github page]
        
        w_skip = self.skip.weight.data # (n_hidden, input_dim)
        w_skip_norms = torch.linalg.norm(w_skip, dim=0) # (1, input_dim)
        skip_gl_pen = w_skip_norms.sum()
        return skip_gl_pen
    
    def train_one_epoch(self, train_loader):
        self.train()
        total_loss = 0
        loss_calculated = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            skip_gl_pen = self.lambda_ * self.skip_GL_penalty()
            if self.config.model.criterion == "MSE":
                target = F.one_hot(target, num_classes=self.config.dataset.n_classes).float() # MSELoss expects one-hot encoded vectors
            loss = self.criterionSum(output, target) + skip_gl_pen
            total_loss += loss.item()
            loss_calculated += len(data)

            if self.config.training.verbose == 2:
                print("loss: ", loss.item())

            loss.backward()
            self.optimizer.step()
            # Proximal stuff
            self.prox(
                    lambda_=self.lambda_ * self.optimizer.param_groups[0]["lr"],
                    M=self.M,
                )

            track_features = self.config.model.hyper.track_features
            if self.config.training.wandb.track and track_features:
                feature_norms = self.get_feature_norms(track_features)
                for idx, feature_norm in zip(track_features, feature_norms):
                    wandb.log({f"feature_{idx}" : feature_norm.item()})


        train_loss = total_loss / loss_calculated
        skip_norms = torch.linalg.norm(self.skip.weight.data, dim=0)
        first_layer_norms = torch.linalg.norm(self.model[0].block[0].weight.data, dim=0)

        print("\nFeatures remaining skip: ", len(skip_norms) - (skip_norms < 1e-2).sum())
        print("Features remaining first: ", len(first_layer_norms) - (first_layer_norms < 1e-2).sum())
        print("Lambda: ", self.lambda_)

        if self.config.training.wandb.track:
            wandb.log({"lambda": self.lambda_})
            wandb.log({"skip_features": len(skip_norms) - (skip_norms < 1e-3).sum()})
            wandb.log({"features": len(first_layer_norms) - (first_layer_norms < self.M*1e-3).sum()})
        # Update lambda
        self.lambda_ *= self.config.model.hyper.lambda_multiplier

        return train_loss
    
    # def train_model(self, train_loader, val_loader):

    #     self.to(self.device)
    #     self.config = self.config

    
    #     if self.config.model.optimizer == "adam":
    #         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model.hyper.lr)

    #     if self.config.model.criterion == "crossentropy":
    #         self.criterion = nn.CrossEntropyLoss()
    #         self.criterionSum = nn.CrossEntropyLoss(reduction='sum')

    #     if self.config.model.criterion == "MSE":
    #         self.criterion = nn.MSELoss()
    #         self.criterionSum = nn.MSELoss(reduction='sum')
        
    #     best_loss = np.inf
    #     no_improve_epochs = 0
        
    #     for epoch in tqdm(range(self.config.model.hyper.epochs), desc="Training", total=self.config.model.hyper.epochs):
    #         train_loss = self.train_one_epoch(train_loader)

    #         if self.config.training.verbose:
    #             print(f'Epoch: {epoch + 1}') 
    #             print(f"Train loss: {train_loss}")
            
    #         if self.config.training.wandb.track:
    #             wandb.log({"train_loss": train_loss})

    #         if val_loader and epoch % self.config.training.evaluate_freq == 0:
    #             val_loss, accuracy = self.evaluate(val_loader)
                
    #             if self.config.training.verbose:
    #                 print(f'Validation loss: {val_loss}') 
    #                 print(f'Accuracy: {accuracy}')
                
    #             if self.config.training.wandb.track:
    #                 wandb.log({"val_loss": val_loss})
    #                 wandb.log({"accuracy": accuracy})
                
    #             if val_loss < best_loss:
    #                 best_loss = val_loss
    #                 self.save_model(self.save_as)
    #                 no_improve_epochs = 0

    #             else:
    #                 no_improve_epochs += 1
    #                 if no_improve_epochs >= self.config.model.hyper.patience:
    #                     print("Early stopping")
    #                     break
                    
    #     # Load the best model
    #     self.load_model(f"classifiers/saved_models/{self.save_as}", map_location=self.device)
    

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

        


    