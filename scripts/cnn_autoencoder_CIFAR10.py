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
    print(f'Running on {config.device}')

# Params for loading data
gen_params = {
        'data_dir': config.upenn_out_dir,
        'csv_dir': config.upenn_dir,
        'modality': ['mods'],
        'n_channels': 5,
        'seed': config.seed,
        'to_augment': False,
        'make_augment': False,
        'to_encode': False,
        'to_slice': False,
        'to_3D_slice': False,
        'use_clinical': False,
        'augment_types': None,
        'batch_sz': config.load_batch_size,
}

#  generate training and testing dataloader
full_dl = datasets.load_upenn_2d_full(gen_params)

#  select modality to train CNN
modality = 'T2' # choose from ['T2', 'FLAIR', 'T1', 'T1GD']
modality_id = helpers.modalities['struct'].index(modality)

#  extracting training and validation images
full_images, full_labels = helpers.extract_images(full_dl, [modality], labels=True)

#  extracting training and validation images
split_idx = int(0.8*full_images.shape[0])

training_images = full_images[0:split_idx]
validation_images = full_images[split_idx:]

#  extracting test images for visualization purposes
test_images = torch.cat([training_images[0:5], validation_images[0:5]],dim=0)
    
#  creating pytorch datasets
training_data = datasets.CustomDataset(training_images, transforms=transforms.Compose([]))

validation_data = datasets.CustomDataset(validation_images, transforms=transforms.Compose([]))

test_data = datasets.CustomDataset(test_images, transforms=transforms.Compose([]))
 
#  training model
model = models.ConvolutionalAutoencoder(models.UPENNAutoencoder(
    models.UPENNEncoder(in_channels=1,out_channels=8,img_len=240,latent_dim=2000),
    models.UPENNDecoder(in_channels=1,out_channels=8,img_len=240,latent_dim=2000)),
    in_channels=1,
    img_len = 240)

log_dict = model.train(nn.MSELoss(), epochs=20, batch_size=128, 
    training_set=training_data, validation_set=validation_data, test_set=test_data)


train_loss = np.asarray(log_dict['training_loss_per_batch'])
val_loss = np.asarray(log_dict['validation_loss_per_batch'])
num_batches = train_loss.shape[0]

plot_data.plot_line(np.arange(num_batches), train_loss, "train loss", 'Training loss per batch', 'batch number', 'training loss', save=True, fname="train_loss")
plot_data.plot_line(np.arange(num_batches), val_loss, "validation loss", 'validation loss per batch', 'batch number', 'validation loss', save=True, fname="val_loss")



