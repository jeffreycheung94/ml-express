# Importing the necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torch 
import ae 

from ae.models.model import DeepAutoencoder
from ae.models.train import train_per_epoch
from cc.dataset.dataset import create_dataset
from cc.dataset.dataset import single_image_inference
import argparse
from torchvision.utils import save_image




def main():
    
    #init parser  
    parser = argparse.ArgumentParser(description='model inputs')
    parser.add_argument('--train', required=False, action='store_true', help='pass in arg if training the model')
    parser.add_argument('--epochs', type=int, required=False, help='number of epochs to train')
    parser.add_argument('--pretrain_weights', required=False, type=str, help='pre-trained weights if inferencing or resuming training')
    parser.add_argument('--batch_size_train', required=False, type=int, default=100, help='desired batch size for training')
    parser.add_argument('--output_dir', required=False, type=str, help='output dir for weights and tb logs')

    #get args
    args = parser.parse_args()
    train = args.train
    pretrain_weights = args.pretrain_weights
    num_epochs = args.epochs
    batch_size = args.batch_size_train
    
    if train:
        transform = torchvision.transforms.Compose([ 
            torchvision.transforms.ToTensor(), 
            # torchvision.transforms.Normalize((0.5), (0.5)) 
        ]) 
        training_loader, validation_loader = create_dataset(256, transform)
        model = DeepAutoencoder()
        criterion = torch.nn.MSELoss() 
        num_epochs = 25
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8) 
        train_per_epoch(training_loader, num_epochs, model, optimizer, criterion)
    else:
        #create model and load weights 
        saved_model = DeepAutoencoder()
        saved_model.load_state_dict(torch.load(pretrain_weights))
        saved_model.eval()
        model_input = single_image_inference('/mnt/c/Users/jeffr/SynologyDrive/Coding/repos/ml-express/scratch/dog.jpg')
        with torch.no_grad():
            result = saved_model(model_input)
        
        #save after auto-encoding
        save_image(result, 'auto_encoded_image.png')
        
if __name__ == '__main__':
    main()


