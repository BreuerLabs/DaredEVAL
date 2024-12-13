import matplotlib.pyplot as plt

def plot_tensor(tensor, save_name):
    arr = tensor.numpy().transpose(1,2,0)
    plt.imshow(arr)
    plt.axis('off')
    plot_path = f"plots/{save_name}.png"
    plt.savefig(plot_path, bbox_inches = 'tight')
    return plt