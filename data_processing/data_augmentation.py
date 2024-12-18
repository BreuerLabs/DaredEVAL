
from torchvision import datasets, transforms


def get_transforms(config, extra_augmentations:list = [], train = True): #! TODO: Make it give training or test augmentations
    # Get dataaugmentations
    image_height = config.dataset.input_size[1]
    image_width  = config.dataset.input_size[2]
    image_size = (image_height, image_width)
    augmentations = []
    
    if config.dataset.resize:
        resize = transforms.Resize(image_size, antialias=True)
        augmentations.append(resize)
    
    if config.dataset.augment_data and train:
        random_resize_crop = transforms.RandomResizedCrop(**config.dataset.transformations.RandomResizedCrop)
        random_horizontal_flip = transforms.RandomHorizontalFlip(**config.dataset.transformations.RandomHorizontalFlip)
        augmentations += [random_resize_crop, random_horizontal_flip]
        
        if config.dataset.input_size[0] == 3:
            color_jitter = transforms.ColorJitter(**config.dataset.transformations.ColorJitter)
            augmentations.append(color_jitter)
    
    if config.dataset.augment_data and not train:
        center_crop = transforms.CenterCrop((image_height, image_width)) #! Already resized?
        augmentations += [center_crop]
        
    if extra_augmentations:
        augmentations += extra_augmentations
            
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
        base_transformations = ([
            # transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        all_transformations = augmentations + base_transformations
    
        transform = transforms.Compose(all_transformations)
    
    elif config.dataset.dataset == "FaceScrub":
        base_transformations = ([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        all_transformations = augmentations + base_transformations
        
        transform = transforms.Compose(all_transformations)
    
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.dataset}")
    
    return transform