# Importing the necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torch 
import ae 

from ae.dataset.mnist_dataset import get_mnist
from ae.models.model import DeepAutoencoder
from ae.core.train import train_per_epoch


plt.rcParams['figure.figsize'] = 15, 10

# Initializing the transform for the dataset 
transform = torchvision.transforms.Compose([ 
	torchvision.transforms.ToTensor(), 
	torchvision.transforms.Normalize((0.5), (0.5)) 
]) 

train_loader, test_loader = get_mnist()
model = DeepAutoencoder()
criterion = torch.nn.MSELoss() 
num_epochs = 25
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 


train = False
test = True
if train==True:
    train_per_epoch(train_loader, num_epochs, model, optimizer, criterion)

elif test==True:
    pass


