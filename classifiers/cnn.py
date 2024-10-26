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
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class CNN(AbstractClassifier, nn.Module):
    
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        
        first_conv = ConvBlock(config.n_channels, n_neurons, kernel_size, stride)
        
        conv_layers = [ConvBlock(n_neurons * 2**i, n_neurons * 2**(i+1), kernel_size, stride) for i in range(n_depth)]
        
        self.model = nn.Sequential(
            first_conv,
            *conv_layers,
            nn.Flatten(),
            nn.Linear(n_neurons * 2**n_depth * 2 * 2, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
            )


if __name__ == "__main__":
    model = CNN(n_classes=10, input_size=32, n_channels=3, n_depth=3, n_neurons=32, kernel_size=3, stride=1)
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    model.train_model(train_loader=trainloader, val_loader=testloader, track=False, optimizer="adam", criterion="crossentropy", 
            lr=0.001, epochs=1, evaluate_freq=1, patience=1, save_as="model.pth", verbose=1, device="cpu")
    
    print(model)
    print(model.device)
    print(model.model)