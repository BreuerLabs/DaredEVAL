from torchvision import datasets
from torch.utils.data import Dataset, random_split
import torch
import os

from data_processing.celeba import CelebA_N_most_common
import data_processing.facescrub as facescrub

def get_datasets(config, train_transform, test_transform):
    """Dynamically loads datasets based on the configuration."""
    
    if config.dataset.dataset == "CIFAR10":
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        
    elif config.dataset.dataset == "MNIST":   
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        
    elif config.dataset.dataset == "FashionMNIST":
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
        
    elif config.dataset.dataset == "FaceScrub":
        # Only runs if it is not downloaded
        facescrub.download() 
        facescrub.process_zip()
        
        full_train_dataset = facescrub.FaceScrub(group = config.dataset.group,
                                                 train = True,
                                                 split_seed = config.training.seed,
                                                 transform = train_transform)
        
        test_dataset = facescrub.FaceScrub(group = config.dataset.group,
                                            train = True,
                                            split_seed = config.training.seed,
                                            transform = test_transform)
        
        
    elif config.dataset.dataset == "CelebA": 
        necessary_files = ["data/celeba/list_eval_partition.txt", 
                           "data/celeba/identity_CelebA.txt",
                           "data/celeba/img_align_celeba",
                           "data/celeba/list_attr_celeba.txt",
                           "data/celeba/list_bbox_celeba.txt",
                           "data/celeba/list_landmarks_align_celeba.txt"]
        
        is_downloaded = all(map(os.path.exists, necessary_files))
        
        # Check if data is downloaded
        if not is_downloaded:
            try:
                _ = datasets.CelebA(root='./data', split='all', target_type = config.dataset.target_type, download=True, transform=transform)
                del _
            
            except Exception as e:
                raise AssertionError("There are problems with torchvision and the automatic download of CelebA. Git issue: https://github.com/pytorch/vision/issues/8204. Temporary fix is to download the files manually and put them in the celeba folder in the data folder.")
            
        
        full_train_dataset = CelebA_N_most_common(train=True,
                                   split_seed=config.training.seed,
                                   transform=train_transform,
                                   root='./data/celeba',
                                   N = config.dataset.n_classes
                                   )
        
        test_dataset = CelebA_N_most_common(train=False,
                                   split_seed=config.training.seed,
                                   transform=test_transform,
                                   root='./data/celeba',
                                   N = config.dataset.n_classes
                                   )

    else:
        raise ValueError(f"Unknown dataset: {config.dataset.dataset}")

    # Split the full training dataset into training and validation sets
    train_size = int((1 - config.training.validation_size) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_dataset = AttackDataset(train_dataset)
    val_dataset = AttackDataset(val_dataset)
    
    #! Maybe also do: test_dataset = AttackDataset(test_dataset)

    return train_dataset, val_dataset, test_dataset


class AttackDataset(Dataset):
    """
    Wrap a subset to add a 'self.targets' attribute. Needed for attacks

    Args:
        subset (torch.utils.data.Subset): A subset of a dataset.
    """
    
    def __init__(self, subset):
        self.subset = subset
        self.targets = torch.tensor([subset.dataset.targets[idx] for idx in subset.indices])

    def __getitem__(self, index):
        return self.subset[index]

    def __len__(self):
        return len(self.subset)

