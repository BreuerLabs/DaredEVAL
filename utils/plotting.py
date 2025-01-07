import matplotlib.pyplot as plt
import torch

def plot_tensor(tensor, save_name):
    arr = tensor.numpy().transpose(1,2,0)
    plt.imshow(arr)
    plt.axis('off')
    plot_path = f"plots/{save_name}.png"
    plt.savefig(plot_path, bbox_inches = 'tight')
    return plt


def plot_lasso_mask(drop_layer_model, plot_name):
    w_first = drop_layer_model.input_defense_layer.weight.data
                    
    n_channels, x_dim, y_dim = drop_layer_model.config.dataset.input_size
    if n_channels == 3:
        w_norms = torch.linalg.norm(w_first, dim=0)
    else: # n_channels == 1
        w_norms = w_first.abs() # does the same thing as norm of dim=0 when n_channels is 1, but this is more readable
        
    w_norms = w_norms.reshape((1, x_dim, y_dim)) # (x_dim, y_dim) -> (1, x_dim, y_dim)

    plt = plot_tensor(w_norms.cpu(), plot_name)
    
    return plt
    
def plot_masked_image(drop_layer_model, tensor_images):
    lasso_mask = drop_layer_model.input_defense_layer
    
    masked_images = lasso_mask(tensor_images)
    
    for i, image in enumerate(masked_images):
        plot_tensor(image, "masked_image_" + i)

def plot_mask_and_masked_images(drop_layer_model, dataset, plot_name):
    
    
    lasso_mask_plot = plot_lasso_mask(drop_layer_model, plot_name)
    
    
    x1, y1 = dataset[0]
    x2, y2 = dataset[1]
    x3, y3 = dataset[2]
    
