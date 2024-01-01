import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO maybe generalize this to take an arg to determine which loss fn to return?
def create_loss_fn():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn
