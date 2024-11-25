
from torchvision import datasets, transforms


def get_transforms(config):
    # Get dataaugmentations
    if config.dataset.augment_data:
        random_resize_crop = transforms.RandomResizedCrop(**config.dataset.transformations.RandomResizedCrop)
        random_horizontal_flip = transforms.RandomHorizontalFlip(**config.dataset.transformations.RandomHorizontalFlip)
        augmentations = [random_resize_crop, random_horizontal_flip]
        
        if config.dataset.input_size[0] == 3:
            color_jitter = transforms.ColorJitter(**config.dataset.transformations.ColorJitter)
            augmentations.append(color_jitter)
            
    else:
        augmentations = []
            
    if config.dataset.dataset == "CIFAR10":
        base_transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        
        all_transformations = augmentations + base_transformations
        transform = transforms.Compose(all_transformations)
        
    elif config.dataset.dataset == "MNIST":
        base_transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        
        all_transformations = augmentations + base_transformations        
        transform = transforms.Compose(all_transformations)
        
        
    elif config.dataset.dataset == "FashionMNIST":
        base_transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        
        all_transformations = augmentations + base_transformations
        
        transform = transforms.Compose(all_transformations)
        
    elif config.dataset.dataset == "CelebA":
        
        image_height = config.dataset.input_size[1]
        image_width  = config.dataset.input_size[2]
        
        base_transformations = ([
            # transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        all_transformations = augmentations + base_transformations
    
        transform = transforms.Compose(all_transformations)
        
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.dataset}")
    
    return transform