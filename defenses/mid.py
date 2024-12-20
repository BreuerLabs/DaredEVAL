import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from classifiers.abstract_classifier import AbstractClassifier

def apply_MID_defense(config, model:AbstractClassifier):
    
    
    class MID(model.__class__):
            
        def __init__(self, config):
            super(MID, self).__init__(config)
            
            
            self.config = config
            # self.original_model = model
            
            
            list_of_layers = list(model.model.children())
            last_layer = list_of_layers[-1]
            # self.feat_dim = 512 * 2 * 2
            self.feat_dim = last_layer.in_features
            self.k = self.feat_dim // 2
            self.n_classes = config.dataset.n_classes
            
            self.model = model
            
            self.model.model.fc = nn.Identity() # removes final classification layer
            
            
            self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
            self.fc_layer = nn.Linear(self.k, self.n_classes)
                
            # self.output_layer = nn.Sequential(nn.BatchNorm2d(self.feat_dim),
            #                                 nn.Dropout(),
            #                                 nn.Flatten(),
            #                                 nn.Linear(self.feat_dim * 4 * 4, self.feat_dim),
            #                                 nn.BatchNorm1d(self.feat_dim))  

            # self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
            
            # self.fc_layer = nn.Sequential(
                                        # nn.Linear(self.k, self.n_classes)
                                            # )
                                    #,nn.Softmax(dim = 1)) #! Probably not needed when using our structure.
                    
            # self.criterion = CrossEntropyLoss #! TODO: Figure out if nescessary to use their implementation
            # self.criterionSum = CrossEntropyLoss 
            
        def forward(self, x, mode="train"):
            feature = self.model(x)

            feature = feature.view(feature.size(0), -1)
            
            statis = self.st_layer(feature)
            
            mu, std = statis[:, :self.k], statis[:, self.k:]
            
            std = F.softplus(std-5, beta=1) #! Parameters?
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
            res = mu + std * eps
            out = self.fc_layer(res)
            
            __, iden = torch.max(out, dim=1)
            iden = iden.view(-1, 1)
        
            return [feature, mu, std, out]
        
        def predict(self, x):
            feature, mu, std, out = self(x)
            
            out = torch.argmax(out, dim=1)
        
            return out

        def debug_forward(self, dataloader):
            self.to(self.device)
            dataset = dataloader.dataset
            x, y = dataset[0]
            x = x.to(self.device)
            
            x = x.unsqueeze(0)
            
            feature, mu, std, out = self(x)
            
            return [feature, mu, std, out]
        
        
        def get_loss(self, output, target):
            ___, mu, std, out_prob = output
            cross_loss = self.criterionSum(out_prob, target)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).sum()
            loss = cross_loss + self.config.defense.beta * info_loss
        
            return loss
        
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
                    
                    _, _, _, out = output
                    
                    pred = torch.argmax(out, dim=1)
                    correct += (pred == target).sum().item()
            
            avg_loss = total_loss / total_instances
            accuracy = correct / total_instances
            
            return avg_loss, accuracy
            
    protected_model = MID(config)
    
    return protected_model


class CrossEntropyLoss(_Loss):
        def forward(self, out, gt, mode="reg"):
            bs = out.size(0)
            loss = - torch.mul(gt.float(), torch.log(out.float() + 1e-7))
            if mode == "dp":
                loss = torch.sum(loss, dim=1).view(-1)
            else:
                loss = torch.sum(loss) / bs
            return loss

class BinaryLoss(_Loss):
    def forward(self, out, gt):
        bs = out.size(0)
        loss = - (gt * torch.log(out.float()+1e-7) + (1-gt) * torch.log(1-out.float()+1e-7))
        loss = torch.mean(loss)
        return loss