# Importing the necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torch 


def get_minst():
	# Downloading the MNIST dataset 
	train_dataset = torchvision.datasets.MNIST( 
		root="./MNIST/train", train=True, 
		transform=torchvision.transforms.ToTensor(), 
		download=True) 

	test_dataset = torchvision.datasets.MNIST( 
		root="./MNIST/test", train=False, 
		transform=torchvision.transforms.ToTensor(), 
		download=True) 

	# Creating Dataloaders from the 
	# training and testing dataset 
	train_loader = torch.utils.data.DataLoader( 
		train_dataset, batch_size=256) 
	test_loader = torch.utils.data.DataLoader( 
		test_dataset, batch_size=256) 

	# Printing 25 random images from the training dataset 
	random_samples = np.random.randint( 
		1, len(train_dataset), (25)) 
	

	return train_loader, test_loader

# for idx in range(random_samples.shape[0]): 
# 	plt.subplot(5, 5, idx + 1) 
# 	plt.imshow(train_dataset[idx][0][0].numpy(), cmap='gray') 
# 	plt.title(train_dataset[idx][1]) 
# 	plt.axis('off') 

# plt.tight_layout() 
# plt.show() 