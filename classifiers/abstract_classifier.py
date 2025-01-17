import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm



class AbstractClassifier(nn.Module):
    """ 
    This is an abstract class for the classifiers. It contains the train, forward, predict, save_model, load_model methods. It should not be used directly. 
    """
    
    def __init__(self, config):
        super(AbstractClassifier, self).__init__()
        self.device = config.training.device
        self.config = config

        if config.model.criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
            self.criterionSum = nn.CrossEntropyLoss(reduction='sum')

        if config.model.criterion == "MSE":
            self.criterion = nn.MSELoss()
            self.criterionSum = nn.MSELoss(reduction='sum')
           

    def init_model(self):
        model = None
        return model
    
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
            self.train_step += 1

            # AbstractClassifiers overwrite this method with anything that needs to be done before the next batch is read
            self.post_batch()

        train_loss = total_loss / loss_calculated

        return train_loss

    def post_batch(self):
        pass
    
    def train_model(self, train_loader, val_loader):
        
        # Distribute model on GPU's
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
    
        
        if self.config.training.save_as:
            self.save_as = self.config.training.save_as + ".pth"
        elif self.config.training.wandb.track:
            self.save_as = wandb.run.name + ".pth" 
        else:
            raise ValueError("Please provide a name to save the model when not using wandb tracking")
        
        if self.config.model.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.config.model.hyper.lr,
                                              betas=(0.9, self.config.model.hyper.beta2),
                                              )

        if self.config.model.lr_scheduler == "MultiStepLR": 
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                             milestones=self.config.model.hyper.milestones,
                                                             gamma=self.config.model.hyper.gamma,
                                                             )

        self.train_step = 0

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
                wandb.log({"train_loss": train_loss, "train_step": self.train_step, "epoch": epoch+1})
                if self.config.model.lr_scheduler:
                    wandb.log({"learning_rate" : self.lr_scheduler.get_last_lr()[0], "train_step" : self.train_step, "epoch": epoch+1})

            if val_loader and epoch % self.config.training.evaluate_freq == 0:
                val_loss, val_accuracy = self.evaluate(val_loader)

                _, train_accuracy = self.evaluate(train_loader)
                del _
                
                if self.config.training.verbose:
                    print(f'Train Accuracy: {train_accuracy}')
                    print(f'Validation loss: {val_loss}') 
                    print(f'Val Accuracy: {val_accuracy}')
                
                if self.config.training.wandb.track:
                    wandb.log({"train_accuracy": train_accuracy, "train_step": self.train_step, "epoch": epoch+1})
                    wandb.log({"val_loss": val_loss, "train_step": self.train_step, "epoch": epoch+1})
                    wandb.log({"val_accuracy": val_accuracy, "train_step": self.train_step, "epoch": epoch+1})
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(self.save_as)
                    no_improve_epochs = 0

                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= self.config.model.hyper.patience:
                        print("Early stopping")
                        break

            if self.config.model.lr_scheduler:
                self.lr_scheduler.step()

            self.post_epoch(epoch)
                 
        # Load the best model
        self.load_model(f"classifiers/saved_models/{self.save_as}", map_location=self.device)
    
    def pre_train(self):
        pass
    
    def post_epoch(self, epoch):
        pass

    
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

    def get_loss(self, output, target): # calculate loss 
        
        loss = self.criterionSum(output, target)
        return loss

    def forward(self, x):
        return self.model(x)

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)
    
    def save_model(self, name):
        path = f"classifiers/saved_models/{name}"
        if isinstance(self.model, nn.DataParallel): # self.model is on DataParallel, need to only save self.model.module
            state = {
                    "model": self.model.module.state_dict(),
                }
        else:
            state= {
                "model": self.model.state_dict(),
            }
        torch.save(state, path)

        
    def load_model(self, file_path, map_location = None):
        if map_location is None:
            state = torch.load(file_path, weights_only=True)
        else:
            state = torch.load(file_path, map_location=map_location, weights_only=True)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state['model'])
        else:
            self.model.load_state_dict(state['model'])
