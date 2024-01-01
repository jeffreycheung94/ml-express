import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import glob

def create_dataset(batch_size, transform):
        
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size, shuffle=False)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    return training_loader, validation_loader

def single_image_inference(image_path):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize((0.5,), (0.5,))])
    img = Image.open(image_path).convert('L')
    img = transform(img)
    
    return img


    
    
