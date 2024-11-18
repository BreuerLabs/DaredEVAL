import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier

class BiDO():

    def __init__(self, config):
        self.config = config
