import torch
import torch.nn as nn
import numpy as np
import wandb

class AbstractClassifier(nn.Module):
    """ 
    This is an abstract class for the classifiers. It contains the train, forward, predict, save_model, load_model methods. It should not be used directly. 
    """
    
    def train_model(self, train_loader, val_loader, track:bool, optimizer:str, criterion:str, 
            lr:float, epochs:int, evaluate_freq:int, patience:int, save_as:str, verbose:int, device:str):
        
        self.to(device)
    
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        
        best_loss = np.inf
        no_improve_epochs = 0
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            loss_calculated = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                loss_calculated += 1

                if verbose == 2:
                    print("loss: ", loss.item())

                loss.backward()
                self.optimizer.step()

            if verbose:
                print(f'Train Epoch: {epoch + 1}, Loss: {total_loss / loss_calculated}')
            
            if track:
                wandb.log({"train_loss": total_loss / loss_calculated})

            if val_loader and epoch % evaluate_freq == 0:
                val_loss = self.get_avg_loss(val_loader)
                print(f'Validation loss: {val_loss}')
                
                if track:
                    wandb.log({"val_loss": val_loss})
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(save_as)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print("Early stopping")
                        return
                    
        # Load the best model
        self.load_model(f"saved_models/{save_as}", map_location=self.device)
        
    def get_avg_loss(self, loader):
        pass

    def forward(self, X):
        self.model(X)

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, file_path, map_location = None):
        if map_location is None:
            state_dict = torch.load(file_path)
            self.load_state_dict(state_dict)
            
        else:
            state_dict = torch.load(file_path, map_location=map_location)

            self.load_state_dict(state_dict)
        
    