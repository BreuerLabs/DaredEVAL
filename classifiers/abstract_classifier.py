import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from classifiers.defense_utils import ElementwiseLinear


class AbstractClassifier(nn.Module):
    """ 
    This is an abstract class for the classifiers. It contains the train, forward, predict, save_model, load_model methods. It should not be used directly. 
    """
    
    def __init__(self, config):
        super(AbstractClassifier, self).__init__()
        self.device = config.training.device
        self.config = config

        if self.config.model.criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
            self.criterionSum = nn.CrossEntropyLoss(reduction='sum')

        if self.config.model.criterion == "MSE":
            self.criterion = nn.MSELoss()
            self.criterionSum = nn.MSELoss(reduction='sum')

        if self.config.defense.name == "drop_layer":
            self.input_defense_layer = self.init_defense_layer()
        else:
            self.input_defense_layer = None

    def init_model(self):
        model = None
        return model


    def init_defense_layer(self):

        if self.config.model.flatten:
            in_features = (self.config.dataset.input_size[1] * self.config.dataset.input_size[2] * self.config.dataset.input_size[0],)
        else:
            in_features = (self.config.dataset.input_size[0], self.config.dataset.input_size[1], self.config.dataset.input_size[2])

        defense_layer = ElementwiseLinear(in_features) # starting point that the implemented classifier can build from

        return defense_layer

    
    def train_one_epoch(self, train_loader):
        
        self.train()
        total_loss = 0
        loss_calculated = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)

            loss = self.get_loss(output, target)

            total_loss += loss.item()            

            loss_calculated += len(data)

            if self.config.training.verbose == 2:
                print("loss: ", loss.item())

            loss.backward()
            self.optimizer.step()
            if self.config.defense.apply_threshold:
               self.apply_threshold()

            track_features = self.config.training.wandb.track_features
            if track_features:
                feature_norms = self.get_feature_norms()
                for idx, feature_norm in zip(track_features, feature_norms):
                    wandb.log({f"feature_{idx}" : feature_norm.item()})


        train_loss = total_loss / loss_calculated

        return train_loss


    def train_model(self, train_loader, val_loader):

        if self.config.training.save_as:
            self.save_as = self.config.training.save_as + ".pth"
        elif self.config.training.wandb.track:
            self.save_as = wandb.run.name + ".pth" 
        else:
            raise ValueError("Please provide a name to save the model when not using wandb tracking")
        
        if self.config.model.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model.hyper.lr)

        self.to(self.device)
        
        best_loss = np.inf
        no_improve_epochs = 0
        
        print("\nTraining using ", self.device)
        
        for epoch in tqdm(range(self.config.model.hyper.epochs), desc="Training", total=self.config.model.hyper.epochs):
            train_loss = self.train_one_epoch(train_loader)

            if self.config.training.verbose:
                print(f'Epoch: {epoch + 1}') 
                print(f"Train loss: {train_loss}")
            
            if self.config.training.wandb.track:
                wandb.log({"train_loss": train_loss})

            if val_loader and epoch % self.config.training.evaluate_freq == 0:
                val_loss, accuracy = self.evaluate(val_loader)
                
                if self.config.training.verbose:
                    print(f'Validation loss: {val_loss}') 
                    print(f'Accuracy: {accuracy}')
                
                if self.config.training.wandb.track:
                    wandb.log({"val_loss": val_loss})
                    wandb.log({"accuracy": accuracy})
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(self.save_as)
                    no_improve_epochs = 0

                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= self.config.model.hyper.patience:
                        print("Early stopping")
                        break
                    
        # Load the best model
        self.load_model(f"classifiers/saved_models/{self.save_as}", map_location=self.device)
    
    def evaluate(self, loader):
        self.to(self.device)
        self.eval()
        
        correct = 0
        total_loss = 0
        total_instances = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)

                loss = self.get_loss(output, target)

                total_loss += loss.item()
                total_instances += len(data)
                
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()
        
        avg_loss = total_loss / total_instances
        accuracy = correct / total_instances
        
        return avg_loss, accuracy

    def get_loss(self, output, target): # calculate loss with whatever penalties added 
        loss = self.criterionSum(output, target)
        if self.config.defense.penalty == "lasso": # normal lasso on first layer weights
                lasso_pen = self.config.defense.lasso_lambda * self.lasso_penalty()
                loss = loss + lasso_pen
        
        return loss

    def forward(self, x):
        if self.config.model.flatten:
            batch_size = x.shape[0]
            x = torch.reshape(x, (batch_size, -1))

        if self.input_defense_layer is not None:
            x = self.input_defense_layer(x)

        return self.model(x)

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)
    
    def save_model(self, name):
        path = f"classifiers/saved_models/{name}"
        torch.save(self.state_dict(), path)
        
    def load_model(self, file_path, map_location = None):
        if map_location is None:
            state_dict = torch.load(file_path, weights_only=True)
            self.load_state_dict(state_dict)
            
        else:
            state_dict = torch.load(file_path, map_location=map_location, weights_only=True)

            self.load_state_dict(state_dict)

    def apply_threshold(self):
        if self.config.defense.name == "drop_layer":
            thresh = self.config.defense.lasso_threshold
            current_w_first = self.input_defense_layer.weight.data
            current_w_norms = torch.linalg.norm(current_w_first, dim=0)
            below_threshold = current_w_norms <= thresh  # which features we will zero out
            new_w_first = current_w_first * ~below_threshold # (n_channels, x_dim, y_dim) x (x_dim, y_dim) = (n_channels, x_dim, y_dim)
            self.input_defense_layer.weight.data = new_w_first
        else:
            assert 0 == 1, f"apply_threshold not yet implemented for defense {self.config.defense.name}"

    def lasso_penalty(self): # Lasso penalty on one-dimensional weights
        
        w_first = self.input_defense_layer.weight
        w_norms = torch.linalg.norm(w_first, dim=0) # takes L2 norm over n_channels. Performs abs() when n_channels=1, combines RGB values of pixels (group lasso) when n_channels=3
        lasso_pen = w_norms.sum()
        return lasso_pen
    
    def get_feature_norms(self):
        w_first = self.input_defense_layer.weight.data
        w_norms = torch.linalg.norm(w_first, dim=0) # takes L2 norm over n_channels
        feature_idxs = OmegaConf.to_object(self.config.training.wandb.track_features)
        
        if isinstance(feature_idxs[0], list): # like drop_layer, one weight for each feature index, norm is just abs. value
            w_features = [w_norms[*feature_idx] for feature_idx in feature_idxs]
            return [torch.abs(w_feature) for w_feature in w_features]
        
        elif isinstance(feature_idxs[0], int):
            if len(w_first.shape) == 1: # just one weight for each feature index, norm is just abs. value
                w_features = [w_norms[feature_idx] for feature_idx in feature_idxs]
                return [torch.abs(w_feature) for w_feature in w_features]
        else:
            assert 0 == 1, f"feature_idxs[0] is type {type(feature_idxs[0])}, not int or list"
            
            # elif len(w_first.shape) == 2: # like SGLNN, take L2 norm of all weights associated with each feature index
            #     w_features = [w_first[:, feature_idx] for feature_idx in feature_idxs]
            #     return [torch.linalg.norm(w_feature) for w_feature in w_features]


