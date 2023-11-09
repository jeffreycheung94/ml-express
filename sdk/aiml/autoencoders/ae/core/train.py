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
            
        # Initializing variable for storing 
        # loss 
        running_loss = 0
        
        # Iterating over the training dataset 
        for batch in train_loader: 
                
            # Loading image(s) and 
            # reshaping it into a 1-d vector 
            img, _ = batch 
            img = img.reshape(-1, 28*28) 
            
            # Generating output 
            out = model(img) 
            
            # Calculating loss 
            loss = criterion(out, img) 
            
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
            torch.save(model.state_dict(), 'best_mode.pth') 	
        # Storing useful images and 
        # reconstructed outputs for the last batch 
        outputs[epoch+1] = {'img': img, 'out': out}
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, running_loss))

# # Plotting the training loss 
# plt.plot(range(1,num_epochs+1),train_loss) 
# plt.xlabel("Number of epochs") 
# plt.ylabel("Training Loss") 
# plt.show()

