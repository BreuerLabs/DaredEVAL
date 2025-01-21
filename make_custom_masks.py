import torch

from utils.plotting import plot_tensor

def make_custom_mask(input_size=(3,224,224), type="upper-half", save_name=None, root_save_path="./classifiers/saved_masks/"):
    if type == 'upper-half': # mask out the upper half of the image
        mask = torch.ones(input_size)
        midpoint = input_size[1] // 2
        
        mask[:, :midpoint, :] = 0
    elif type == 'lower-half':
        mask = torch.ones(input_size)
        midpoint = input_size[1] // 2

        mask[:, midpoint:, :] = 0

    else:
        raise Error(f"Type {type} is not defined")

    if save_name is None:
        save_name = f"{type}-mask"
    plt = plot_tensor(mask, save_name=save_name)
    plt.show() # check that mask is right

    # save mask tensor
    torch.save(mask, f"{root_save_path}/{save_name}.pt")


if __name__ == "__main__":
    make_custom_mask(type='upper-half')
    make_custom_mask(type='lower-half')
