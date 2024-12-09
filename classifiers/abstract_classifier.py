import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm

from classifiers.drop_layer import ElementwiseLinear, lasso_penalty, get_feature_norms


class AbstractClassifier(nn.Module):
    """ 
    This is an abstract class for the classifiers. It contains the train, forward, predict, save_model, load_model methods. It should not be used directly. 
    """
    
    def init_model(self, config):
        model = None
        self.device = config.training.device
        self.config = config

        if config.model.criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
            self.criterionSum = nn.CrossEntropyLoss(reduction='sum')

        if config.model.criterion == "MSE":
            self.criterion = nn.MSELoss()
            self.criterionSum = nn.MSELoss(reduction='sum')


        if config.model.drop_layer:
            if config.model.flatten:
                in_features = (config.dataset.input_size[1] * config.dataset.input_size[2] * config.dataset.input_size[0],)
            else:
                in_features = (config.dataset.input_size[0], config.dataset.input_size[1], config.dataset.input_size[2])

            model = ElementwiseLinear(in_features) # starting point that the implemented classifier can build from

        return model

    
    def train_one_epoch(self, train_loader):
        
        config = self.config
        self.train()
        total_loss = 0
        loss_calculated = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterionSum(output, target)
            if config.model.penalty == "lasso": # normal lasso on first layer weights
                lasso_pen = config.model.hyper.lasso_lambda * lasso_penalty(self.model)
                loss += lasso_pen

            total_loss += loss.item()            
            # loss = self.criterion(output, target)
            # total_loss += loss.item() * len(data)
            loss_calculated += len(data)

            if config.training.verbose == 2:
                print("loss: ", loss.item())

            loss.backward()
            self.optimizer.step()

            track_features = self.config.training.wandb.track_features
            if track_features:
                feature_norms = get_feature_norms(self.model, track_features)
                for idx, feature_norm in zip(track_features, feature_norms):
                    wandb.log({f"feature_{idx}" : feature_norm.item()})


        train_loss = total_loss / loss_calculated

        return train_loss


    def train_model(self, train_loader, val_loader):
        config = self.config

        if config.training.save_as:
            self.save_as = config.training.save_as + ".pth"
        elif config.training.wandb.track:
            self.save_as = wandb.run.name + ".pth" 
        else:
            raise ValueError("Please provide a name to save the model when not using wandb tracking")
        
        if config.model.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.model.hyper.lr)

        self.to(self.device)
        
        best_loss = np.inf
        no_improve_epochs = 0
        
        print("\nTraining using ", self.device)
        
        for epoch in tqdm(range(config.model.hyper.epochs), desc="Training", total=config.model.hyper.epochs):
            train_loss = self.train_one_epoch(train_loader)

            if config.training.verbose:
                print(f'Epoch: {epoch + 1}') 
                print(f"Train loss: {train_loss}")
            
            if config.training.wandb.track:
                wandb.log({"train_loss": train_loss})

            if val_loader and epoch % config.training.evaluate_freq == 0:
                val_loss, accuracy = self.evaluate(val_loader)
                
                if config.training.verbose:
                    print(f'Validation loss: {val_loss}') 
                    print(f'Accuracy: {accuracy}')
                
                if config.training.wandb.track:
                    wandb.log({"val_loss": val_loss})
                    wandb.log({"accuracy": accuracy})
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(self.save_as)
                    no_improve_epochs = 0

                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= config.model.hyper.patience:
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

                loss = self.criterionSum(output, target).item()
                if self.config.model.penalty == "lasso": # normal lasso on first layer weights
                    lasso_pen = self.config.model.hyper.lasso_lambda * lasso_penalty(self.model)
                    loss += lasso_pen

                total_loss += loss
                total_instances += len(data)
                
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()
        
        avg_loss = total_loss / total_instances
        accuracy = correct / total_instances
        
        return avg_loss, accuracy

    def forward(self, x):
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


