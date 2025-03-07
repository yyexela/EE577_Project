#!/home/ee577/miniconda3/envs/ee577/bin/python
# From https://www.digitalocean.com/community/tutorials/convolutional-autoencoder

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
from pathlib import Path

pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)

from src import *

# Load config file
config = global_config.config


#  configuring device
if torch.cuda.is_available():
    config.device = 'cuda:2' # use GPU #3
    print('Running on the GPU')
else:
    config.device = 'cpu'
    print('Running on the CPU')

#  loading training data
training_set = Datasets.CIFAR10(root='../Datasets/CIFAR10/', download=True,
                                transform=transforms.ToTensor())

#  loading validation data
validation_set = Datasets.CIFAR10(root='../Datasets/CIFAR10/', download=True, train=False,
                                  transform=transforms.ToTensor()) 
    
#  extracting training images
training_images = [x for x in training_set.data]

#  extracting validation images
validation_images = [x for x in validation_set.data]

#  extracting test images for visualization purposes
test_images = helpers.extract_each_class(validation_set) 
    
#  creating pytorch datasets
training_data = datasets.CustomDataset(training_images, transforms=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

validation_data = datasets.CustomDataset(validation_images, transforms=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

test_data = datasets.CustomDataset(test_images, transforms=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
 
#  training model
model = models.ConvolutionalAutoencoder(models.CIFAR10Autoencoder(models.CIFAR10Encoder(), models.CIFAR10Decoder()))

log_dict = model.train(nn.MSELoss(), epochs=20, batch_size=64, 
    training_set=training_data, validation_set=validation_data, test_set=test_data)



train_loss = np.asarray(log_dict['training_loss_per_batch'])
val_loss = np.asarray(log_dict['validation_loss_per_batch'])
num_batches = train_loss.shape[0]

print(train_loss.shape)
print(np.arange(num_batches).shape)

plot_data.plot_line(np.arange(num_batches), train_loss, "train loss", 'Training loss per batch', 'batch number', 'training loss', fname="train_loss", save=True)
plot_data.plot_line(np.arange(num_batches), val_loss, "validation loss", 'validation loss per batch', 'batch number', 'validation loss', fname="val_loss", save=True)



