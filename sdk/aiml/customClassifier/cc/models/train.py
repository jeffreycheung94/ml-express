import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_epoch(epoch_index, tb_writer, training_loader, validation_loader, optimizer, loss_fn, model):
    running_loss = 0.
    last_loss = 0.
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair

        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 0:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss