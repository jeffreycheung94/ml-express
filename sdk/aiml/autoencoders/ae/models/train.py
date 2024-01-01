# Importing the necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torch 

# List that will store the training loss 
train_loss = [] 

# Dictionary that will store the 
# different images and outputs for 
# various epochs 
outputs = {} 

def train_per_epoch(train_loader, num_epochs, model, optimizer, criterion):
    batch_size = len(train_loader) 
    # Training loop starts 
    for epoch in range(num_epochs): 
        # Initializing variable for storing loss
        running_loss = 0
        # Iterating over the training dataset 
        for batch in train_loader: 
            out = model(batch[0]) 
            loss = criterion(out, batch[0]) 
            print(loss)
            
            # Updating weights according 
            # to the calculated loss 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            # Incrementing loss 
            running_loss += loss.item()
            
        # Averaging out loss over entire batch 
        running_loss /= batch_size 
        train_loss.append(running_loss)
        if running_loss < train_loss[epoch-1]:
            print('saving new best model')
            torch.save(model.state_dict(), 'best_model.pth') 	


