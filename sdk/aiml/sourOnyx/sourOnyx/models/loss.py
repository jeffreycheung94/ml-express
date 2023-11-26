import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO maybe generalize this to take an arg to determine which loss fn to return?
def create_loss_fn():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn


# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
# dummy_outputs = torch.rand(4, 10)
# # Represents the correct class among the 10 being tested
# dummy_labels = torch.tensor([1, 5, 3, 7])

# print(dummy_outputs)
# print(dummy_labels)

# loss = loss_fn(dummy_outputs, dummy_labels)
# print('Total loss for this batch: {}'.format(loss.item()))