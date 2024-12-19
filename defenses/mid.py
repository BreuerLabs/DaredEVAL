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
            self.feature = model.features
            self.feat_dim = 512 * 2 * 2 #! TODO: Read from model
            self.k = self.feat_dim // 2
            self.n_classes = config.dataset.n_classes
            
            self.st_layer = nn.Linear(self.feat_dim, self.k * 2) 
            self.fc_layer = nn.Linear(self.k, self.n_classes)
            
            # self.criterion = CrossEntropyLoss #! TODO: Figure out if nescessary to use their implementation
            # self.criterionSum = CrossEntropyLoss 
            
        def debug_forward(self, dataloader):
            dataset = dataloader.dataset
            x, y = dataset[0]
            x, y = x.to(self.device), y.to(self.device)
            
            feature, mu, std, out = self(x)
            
            return feature, mu, std, out
            
            
            
            
        def train_one_epoch(self, train_loader):
            self.train()
            total_loss = 0
            loss_calculated = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                bs = data.size(0)
                target = target.view(-1)
                
                ___, mu, std, out_prob = model(data)
                cross_loss = self.criterion(out_prob, target)
                # loss = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
                loss = cross_loss + self.config.defense.beta * info_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                out_target = torch.argmax(out_prob, dim=1).view(-1)
                ACC += torch.sum(target == out_target).item()
                total_loss += loss.item() * bs
                cnt += bs                            #? What is this??
            
            train_loss = total_loss / loss_calculated

            return train_loss
                
        def forward(self, x, mode="train"):
            feature = self.feature(x)
            feature = feature.view(feature.size(0), -1)
            
            statis = self.st_layer(feature)
            mu, std = statis[:, :self.k], statis[:, self.k:]
            
            std = F.softplus(std-5, beta=1)
            eps = torch.FloatTensor(std.size()).normal_().cuda()
            res = mu + std * eps
            out = self.fc_layer(res)
        
            return [feature, mu, std, out]
        
        def predict(self, x):
            feature = self.feature(x)
            feature = feature.view(feature.size(0), -1)
            statis = self.st_layer(feature)
            mu, std = statis[:, :self.k], statis[:, self.k:]
            
            std = F.softplus(std-5, beta=1)
            eps = torch.FloatTensor(std.size()).normal_().cuda()
            res = mu + std * eps
            out = self.fc_layer(res)
        
            return out
        
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

    return MID