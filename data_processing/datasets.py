from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg

from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset, random_split
import torch


import numpy as np
import pandas as pd

from functools import partial

import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
import os
from collections import Counter
from data_processing.data_augmentation import get_transforms


def get_datasets(config):
    """Dynamically loads datasets based on the configuration."""
    
    transform = get_transforms(config)
    
    if config.dataset.dataset == "CIFAR10":
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    elif config.dataset.dataset == "MNIST":   
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    elif config.dataset.dataset == "FashionMNIST":
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        
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
            
            except Exception as e:
                raise AssertionError("There are problems with torchvision and the automatic download of CelebA. Git issue: https://github.com/pytorch/vision/issues/8204. Temporary fix is to download the files manually and put them in the celeba folder in the data folder.")
            
        
        full_train_dataset = CelebA_N_most_common(train=True,
                                   split_seed=config.training.seed,
                                   transform=transform,
                                   root='./data/celeba',
                                   N = config.dataset.n_classes
                                   )
        
        test_dataset = CelebA_N_most_common(train=False,
                                   split_seed=config.training.seed,
                                   transform=transform,
                                   root='./data/celeba',
                                   N = config.dataset.n_classes
                                   )
        
        # test_dataset = datasets.CelebA(root='./data', split='test', target_type = target, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {config.dataset.dataset}")

    # Split the full training dataset into training and validation sets
    train_size = int((1 - config.training.validation_size) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


class CelebA_N_most_common(Dataset):
    def __init__(self,
                 train,
                 split_seed=42,
                 transform=None,
                 root='data/celeba',
                 N = 1000,
                 download: bool = False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root,
                        split='all',
                        target_type="identity")
        celeba.targets = celeba.identity

        # Select the 1,000 most frequent celebrities from the dataset
        targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(targets).items(),
                   key=lambda item: item[1],
                   reverse=True))
        sorted_targets = list(ordered_dict.keys())[:N]

        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(targets, sorted_targets))[0]
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Set transformations
        self.transform = transform
        target_mapping = {
            sorted_targets[i]: i
            for i in range(len(sorted_targets))
        }

        self.target_transform = transforms.Lambda(lambda x: target_mapping[x])

        # Split dataset
        if train:
            self.dataset = Subset(celeba, train_idx)
            train_targets = np.array(targets)[train_idx]
            self.targets = [self.target_transform(t) for t in train_targets]
            self.name = 'CelebA1000_train'
        else:
            self.dataset = Subset(celeba, test_idx)
            test_targets = np.array(targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]
            self.name = 'CelebA1000_test'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CustomCelebA(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
            self,
            root: str,
            split: str = "all",
            target_type: Union[List[str], str] = "identity",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(CustomCelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pd.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pd.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pd.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pd.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pd.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor') # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = os.path.join(self.root, "img_align_celeba", self.filename[index])
        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        X = PIL.Image.open(file_path)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)





# def get_subset(dataset, N, seed):
    
#     """
#     Processes the CelebA dataset to return training and test subsets containing only the top 1000 most common classes.

#     Parameters:
#     ----------
#     dataset : Dataset
#         The input CelebA dataset containing images and corresponding labels. It is assumed that the dataset
#         includes a 'label' or equivalent field indicating the class of each sample.

#     Returns:
#     -------
#     train_dataset : Dataset
#         A processed training subset of the CelebA dataset containing only the top 1000 most common classes.
    
#     test_dataset : Dataset
#         A processed testing subset of the CelebA dataset containing only the top 1000 most common classes.

#     """
#     print("\n THIS LINE IS BEING RUN!!!")
#     dataset.targets = dataset.identity

#     # Select the 1,000 most frequent celebrities from the dataset
#     targets = np.array([t.item() for t in dataset.identity])
#     ordered_dict = dict(
#         sorted(Counter(targets).items(),
#                 key=lambda item: item[1],
#                 reverse=True))
#     sorted_targets = list(ordered_dict.keys())[:N]

#     # Select the corresponding samples for train and test split
#     indices = np.where(np.isin(targets, sorted_targets))[0]
#     np.random.seed(seed)
#     np.random.shuffle(indices)
#     training_set_size = int(0.9 * len(indices))
#     train_idx = indices[:training_set_size]
#     test_idx = indices[training_set_size:]

#     # Assert that there are no overlapping datasets
#     assert len(set.intersection(set(train_idx), set(test_idx))) == 0

#     # # Set transformations
#     # self.transform = transform
#     target_mapping = {
#         sorted_targets[i]: i
#         for i in range(len(sorted_targets))
#     }

#     target_transform = transforms.Lambda(lambda x: target_mapping[x])

#     # Split dataset
    
#     train_dataset = Subset(dataset, train_idx)
#     train_targets = np.array(targets)[train_idx]
#     train_dataset.targets = [target_transform(t) for t in train_targets]
#     # self.name = 'CelebA1000_train'
    
#     test_dataset = Subset(dataset, test_idx)
#     test_targets = np.array(targets)[test_idx]
#     test_dataset.targets = [target_transform(t) for t in test_targets]
#     # self.name = 'CelebA1000_test'

#     return train_dataset, test_dataset
    

