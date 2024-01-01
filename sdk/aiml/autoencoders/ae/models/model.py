# Importing the necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torch 

# Creating a DeepAutoencoder class 
class DeepAutoencoder(torch.nn.Module): 
	def __init__(self): 
		super().__init__()		 
		self.encoder = torch.nn.Sequential( 
			torch.nn.Linear(28, 28), 
			torch.nn.ReLU(), 
		) 
		
		self.decoder = torch.nn.Sequential( 
			torch.nn.Linear(28,  28), 
			torch.nn.Sigmoid() 
		) 

	def forward(self, x): 
		encoded = self.encoder(x) 
		decoded = self.decoder(encoded) 
		return decoded 






