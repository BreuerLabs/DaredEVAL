import torch
import torch.nn as nn
from classifiers.abstractClassifier import AbstractClassifier
import torchvision
from torchvision import transforms


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.block = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            self.pool,
                )
        
    def forward(self, x):
        return self.block(x)

class CNN(AbstractClassifier, nn.Module):
    
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        
        first_conv = ConvBlock(config.n_channels, config.n_neurons, config.kernel_size, config.stride)
        
        conv_layers = [ConvBlock(config.n_neurons * 2**i, config.n_neurons * 2**(i+1), config.kernel_size, config.stride) for i in range(n_depth)]
        
        self.model = nn.Sequential(
            first_conv,
            *conv_layers,
            nn.Flatten(),
            nn.Linear(config.n_neurons * 2**config.n_depth * 2 * 2, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, config.n_classes)
            )
