import matplotlib.pyplot as plt

def plot_tensor(tensor, save_name):
    arr = tensor.numpy().transpose(1,2,0)
    plt.imshow(arr)
    plt.axis('off')
    plt.savefig(f"plots/{save_name}.png", bbox_inches = 'tight')