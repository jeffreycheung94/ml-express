import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse

#custom includes
from cc.models.model import ClassifierModel 
from cc.models.train import train_one_epoch 
from cc.models.loss import create_loss_fn 
from cc.dataset.dataset import create_dataset
from pathlib import Path

def main():

    #TODO move arg parser to its own module / method
    #init parser  
    parser = argparse.ArgumentParser(description='model inputs')
    parser.add_argument('--train', required=False, action='store_true', help='pass in arg if training the model')
    parser.add_argument('--epochs', type=int, required=False, help='number of epochs to train')
    parser.add_argument('--weights', required=False, action='store_true', help='pre-trained weights if inferencing or resuming training')
    parser.add_argument('--batch_size_train', required=True, type=int, default=100, help='desired batch size for training')
    parser.add_argument('--output_dir', required=True, type=str, help='output dir for weights and tb logs')


    #get args
    args = parser.parse_args()
    train = args.train
    pretrain_weights = args.weights
    num_epochs = args.epochs
    batch_size = args.batch_size_train
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    #intialize model
    model = ClassifierModel()

    #initalize dataloaders, this downloads fashion mnist from torchvision
    training_loader, validation_loader = create_dataset(batch_size)

    if train: 
        if torch.cuda.is_available():
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
            model.to(device)
            print('training with gpu!')
        
        #initalize optimizer and loss 
        #TODO convert optimizer to a method
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_fn = create_loss_fn()
        best_vloss = 1_000_000.

        # Initializing in a separate cell so we can easily add more epochs to the same run
        writer = SummaryWriter((output_dir))

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch + 1))

            model.train(True)
            avg_loss = train_one_epoch(epoch, writer, training_loader, validation_loader, optimizer, loss_fn, model)
            running_vloss = 0.0
            model.eval()
            #TODO convert this to a eval function
            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(model.state_dict(), output_dir.joinpath('best_model.pth'))
            epoch += 1
    else:
        #create model and load weights 
        saved_model = ClassifierModel()
        saved_model.load_state_dict(torch.load(pretrain_weights))
        
        
        saved_model.eval()
        # with torch.no_grad():
        #     # for i, vdata in enumerate(validation_loader):
        #     #     vinputs, vlabels = vdata
        #     #     voutputs = model(vinputs)
        #     #     vloss = loss_fn(voutputs, vlabels)

if __name__ == '__main__':
    main()