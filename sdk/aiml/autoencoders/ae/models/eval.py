# Importing the necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torch 

# Plotting is done on a 7x5 subplot 
# Plotting the reconstructed images 

# Initializing subplot counter 
counter = 1

# Plotting reconstructions 
# for epochs = [1, 5, 10, 50, 100] 
# epochs_list = [1, 5, 10, 50, 100] 
epochs_list = [1, 5, 10, 25] 


# # Iterating over specified epochs 
# for val in epochs_list: 
	
# 	# Extracting recorded information 
# 	temp = outputs[val]['out'].detach().numpy() 
# 	title_text = f"Epoch = {val}"
	
# 	# Plotting first five images of the last batch 
# 	for idx in range(5): 
# 		plt.subplot(7, 5, counter) 
# 		plt.title(title_text) 
# 		plt.imshow(temp[idx].reshape(28,28), cmap= 'gray') 
# 		plt.axis('off') 
		
# 		# Incrementing the subplot counter 
# 		counter+=1

# # Plotting original images 

# # Iterating over first five 
# # images of the last batch 
# for idx in range(5): 
	
# 	# Obtaining image from the dictionary 
# 	val = outputs[10]['img'] 
	
# 	# Plotting image 
# 	plt.subplot(7,5,counter) 
# 	plt.imshow(val[idx].reshape(28, 28), 
# 			cmap = 'gray') 
# 	plt.title("Original Image") 
# 	plt.axis('off') 
	
# 	# Incrementing subplot counter 
# 	counter+=1

# plt.tight_layout() 
# plt.show()

