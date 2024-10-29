import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm

class AbstractClassifier(nn.Module):
    """ 
    This is an abstract class for the classifiers. It contains the train, forward, predict, save_model, load_model methods. It should not be used directly. 
    """
    
    def init_model(self, config):
        model = None
        self.device = config.training.device
        #self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        return model

    def train_model(self, train_loader, val_loader):

        self.to(self.device)
        config = self.config

    
        if config.model.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.model.hyper.lr)

        if config.model.criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
            self.criterionSum = nn.CrossEntropyLoss(reduction='sum')
        
        best_loss = np.inf
        no_improve_epochs = 0
        
        for epoch in tqdm(range(config.model.hyper.epochs), desc="Training", total=config.model.hyper.epochs):
            self.train()
            total_loss = 0
            loss_calculated = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * len(data)
                loss_calculated += len(data)

                if config.training.verbose == 2:
                    print("loss: ", loss.item())

                loss.backward()
                self.optimizer.step()

            train_loss = total_loss / loss_calculated
            if config.training.verbose:
                print(f'Epoch: {epoch + 1}') 
                print(f"Train loss: {train_loss}")
            
            if config.training.wandb.track:
                wandb.log({"train_loss": train_loss})

            if val_loader and epoch % config.training.evaluate_freq == 0:
                val_loss, accuracy = self.evaluate(val_loader)
                print(f'Validation loss: {val_loss}') 
                print(f'Accuracy: {accuracy}')
                
                if config.training.wandb.track:
                    wandb.log({"val_loss": val_loss})
                    wandb.log({"accuracy": accuracy})
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(config.training.save_as)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= config.model.hyper.patience:
                        print("Early stopping")
                        return
                    
        # Load the best model
        self.load_model(f"classifiers/saved_models/{config.training.save_as}", map_location=self.device)
        
    def get_avg_loss(self, loader):
        self.eval()
        total_loss = 0
        total_instances = 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            total_loss += self.criterionSum(output, target).item()
            total_instances += len(data)
        
        avg_loss = total_loss / total_instances
        
        return avg_loss
    
    def evaluate(self, loader):
        self.eval()
        
        correct = 0
        total_loss = 0
        total_instances = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterionSum(output, target).item()
                total_instances += len(data)
                
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()
        
        avg_loss = total_loss / total_instances
        accuracy = correct / total_instances
        
        return avg_loss, accuracy

    def forward(self, X):
        self.model(X)

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)
    
    def save_model(self, name):
        path = f"classifiers/saved_models/{name}"
        torch.save(self.state_dict(), path)
        
    def load_model(self, file_path, map_location = None):
        if map_location is None:
            state_dict = torch.load(file_path)
            self.load_state_dict(state_dict)
            
        else:
            state_dict = torch.load(file_path, map_location=map_location)

            self.load_state_dict(state_dict)
        
    