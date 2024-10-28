import torch
import torch.nn as nn
# from classifiers.abstractClassifier import AbstractClassifier
import torchvision
from torchvision import transforms

from lassonet import LassoNetClassifierCV

lsn = LassoNetClassifierCV()

print(type(lsn))