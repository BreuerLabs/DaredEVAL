import torch

has_cuda = torch.cuda.is_available()
print("Cuda available:", has_cuda)