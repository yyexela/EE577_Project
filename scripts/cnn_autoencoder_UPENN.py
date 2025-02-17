#!/home/ee577/miniconda3/envs/ee577/bin/python
# From https://www.digitalocean.com/community/tutorials/convolutional-autoencoder

import os
import sys
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torchvision.datasets as Datasets
import torchvision.transforms as transforms

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


# Params for loading data
gen_params = {
    'data_dir': os.path.join(config.upenn_dir, 'numpy_conversion_struct_channels'),
    'csv_dir': os.path.join(config.upenn_dir),
    'modality': ['mods'],
    'dim': (70,86,86),
    'n_channels': 4,
    'n_classes': 7,
    'seed': config.seed,
    'to_augment': False,
    'make_augment': False,
    'to_encode': False,
    'to_slice': True,
    'to_3D_slice': False,
    'n_slices': 1,
    'use_clinical': False,
    'augment_types': None,
    'batch_sz': 2,
}

#  generate training and testing dataloader
training_dl, validation_dl = datasets.load_upenn_2d_struct(gen_params)

#  select modality to train CNN
modality = 'T2' # choose from ['T2', 'FLAIR', 'T1', 'T1GD']
modality_id = helpers.modalities['struct'].index(modality)

for batch, (X, y) in enumerate(training_dl):
    pass

#  extracting training and validation images
training_images = helpers.extract_images(training_dl, modality_id)
validation_images = helpers.extract_images(validation_dl, modality_id)

#  extracting test images for visualization purposes
test_images = validation_images[0:10]
    
#  creating pytorch datasets
training_data = datasets.CustomCIFAR10(training_images, transforms=transforms.Compose([]))

validation_data = datasets.CustomCIFAR10(validation_images, transforms=transforms.Compose([]))

test_data = datasets.CustomCIFAR10(test_images, transforms=transforms.Compose([]))
 
#  training model
model = models.ConvolutionalAutoencoder(models.Autoencoder(
    models.Encoder(in_channels=1,out_channels=8,img_len=86),
    models.Decoder(in_channels=1,out_channels=8,img_len=86)),
    in_channels=1,
    img_len = 86)

log_dict = model.train(nn.MSELoss(), epochs=20, batch_size=64, 
    training_set=training_data, validation_set=validation_data, test_set=test_data)

train_loss = np.asarray(log_dict['training_loss_per_batch'])
val_loss = np.asarray(log_dict['validation_loss_per_batch'])
num_batches = train_loss.shape[0]

plot_data.plot_line(np.arange(num_batches), train_loss, "train loss", 'Training loss per batch', 'batch number', 'training loss', save=True, fname="train_loss")
plot_data.plot_line(np.arange(num_batches), val_loss, "validation loss", 'validation loss per batch', 'batch number', 'validation loss', save=True, fname="val_loss")
