import torch
import torch.nn as nn
import torch.optim as optim
from torch.datasets import datasets, transforms
from torch.utils.data import DataLoader

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Noramalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)) # each of the 3 channels should have mean and std as 0.5
])

# Load the data
train_data = datasets.CIFAR100(root = './dir', train = True, download = True, transform = transform) #"." means current directory; we are downloading the dataset here
test_data = datasets.CIFAR100(root = './dir', train = False, download = True, transform = transform)


