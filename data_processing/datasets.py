
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import Subset
import torch
from torch.utils.data import random_split
import numpy as np
from collections import Counter
import numpy as np
from collections import Counter

def get_datasets(config):
    """Dynamically loads datasets based on the configuration."""
    
    if config.dataset.dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    elif config.dataset.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    elif config.dataset.dataset == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        
    elif config.dataset.dataset == "CelebA":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        full_train_dataset = datasets.CelebA(root='./data', split='all', target_type = config.dataset.target_type, download=False, transform=transform)
        
        full_train_dataset, test_dataset = get_subset(dataset = full_train_dataset, 
                            N = config.dataset.N_most_common_classes, 
                            seed = config.training.seed)
        
        # test_dataset = datasets.CelebA(root='./data', split='test', target_type = target, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {config.dataset.dataset}")

    # Split the full training dataset into training and validation sets
    train_size = int((1 - config.training.validation_size) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def get_subset(dataset, N, seed):
    
    """
    Processes the CelebA dataset to return training and test subsets containing only the top 1000 most common classes.

    Parameters:
    ----------
    dataset : Dataset
        The input CelebA dataset containing images and corresponding labels. It is assumed that the dataset
        includes a 'label' or equivalent field indicating the class of each sample.

    Returns:
    -------
    train_dataset : Dataset
        A processed training subset of the CelebA dataset containing only the top 1000 most common classes.
    
    test_dataset : Dataset
        A processed testing subset of the CelebA dataset containing only the top 1000 most common classes.

    """
    dataset.targets = dataset.identity

    # Select the 1,000 most frequent celebrities from the dataset
    targets = np.array([t.item() for t in dataset.identity])
    ordered_dict = dict(
        sorted(Counter(targets).items(),
                key=lambda item: item[1],
                reverse=True))
    sorted_targets = list(ordered_dict.keys())[:N]

    # Select the corresponding samples for train and test split
    indices = np.where(np.isin(targets, sorted_targets))[0]
    np.random.seed(seed)
    np.random.shuffle(indices)
    training_set_size = int(0.9 * len(indices))
    train_idx = indices[:training_set_size]
    test_idx = indices[training_set_size:]

    # Assert that there are no overlapping datasets
    assert len(set.intersection(set(train_idx), set(test_idx))) == 0

    # # Set transformations
    # self.transform = transform
    # target_mapping = {
    #     sorted_targets[i]: i
    #     for i in range(len(sorted_targets))
    # }

    # self.target_transform = T.Lambda(lambda x: target_mapping[x])

    # Split dataset
    
    train_dataset = Subset(dataset, train_idx)
    # train_targets = np.array(targets)[train_idx]
    # targets = [self.target_transform(t) for t in train_targets]
    # self.name = 'CelebA1000_train'
    
    test_dataset = Subset(dataset, test_idx)
    # test_targets = np.array(targets)[test_idx]
    # self.targets = [self.target_transform(t) for t in test_targets]
    # self.name = 'CelebA1000_test'

    return train_dataset, test_dataset
    





def get_subset(dataset, N, seed):
    
    """
    Processes the CelebA dataset to return training and test subsets containing only the top 1000 most common classes.

    Parameters:
    ----------
    dataset : Dataset
        The input CelebA dataset containing images and corresponding labels. It is assumed that the dataset
        includes a 'label' or equivalent field indicating the class of each sample.

    Returns:
    -------
    train_dataset : Dataset
        A processed training subset of the CelebA dataset containing only the top 1000 most common classes.
    
    test_dataset : Dataset
        A processed testing subset of the CelebA dataset containing only the top 1000 most common classes.

    """
    dataset.targets = dataset.identity

    # Select the 1,000 most frequent celebrities from the dataset
    targets = np.array([t.item() for t in dataset.identity])
    ordered_dict = dict(
        sorted(Counter(targets).items(),
                key=lambda item: item[1],
                reverse=True))
    sorted_targets = list(ordered_dict.keys())[:N]

    # Select the corresponding samples for train and test split
    indices = np.where(np.isin(targets, sorted_targets))[0]
    np.random.seed(seed)
    np.random.shuffle(indices)
    training_set_size = int(0.9 * len(indices))
    train_idx = indices[:training_set_size]
    test_idx = indices[training_set_size:]

    # Assert that there are no overlapping datasets
    assert len(set.intersection(set(train_idx), set(test_idx))) == 0

    # # Set transformations
    # self.transform = transform
    # target_mapping = {
    #     sorted_targets[i]: i
    #     for i in range(len(sorted_targets))
    # }

    # self.target_transform = T.Lambda(lambda x: target_mapping[x])

    # Split dataset
    
    train_dataset = Subset(dataset, train_idx)
    # train_targets = np.array(targets)[train_idx]
    # targets = [self.target_transform(t) for t in train_targets]
    # self.name = 'CelebA1000_train'
    
    test_dataset = Subset(dataset, test_idx)
    # test_targets = np.array(targets)[test_idx]
    # self.targets = [self.target_transform(t) for t in test_targets]
    # self.name = 'CelebA1000_test'

    return train_dataset, test_dataset
    


