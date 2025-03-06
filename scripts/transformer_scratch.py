
#!/home/ee577/miniconda3/envs/ee577/bin/python
# ## [CNN Autoencoder](https://www.digitalocean.com/community/tutorials/convolutional-autoencoder)
# From https://www.digitalocean.com/community/tutorials/convolutional-autoencoder

import os
import sys
import torch
import requests
import numpy as np
import torchvision
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel

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
training_dl, validation_dl = datasets.load_upenn_2d_struct(gen_params)

#  select modality to train CNN
modalities = ['T2', 'FLAIR', 'T1', 'T1GD', 'mask']

for batch, (X, y) in enumerate(training_dl):
    pass

#  extracting training and validation images
training_images, training_labels = helpers.extract_images(training_dl, modalities, labels=True)
validation_images, validation_labels = helpers.extract_images(validation_dl, modalities, labels=True)

# Set up dataset
training_data = datasets.CustomLabeledDataset(training_images, training_labels)

validation_data = datasets.CustomLabeledDataset(validation_images, validation_labels)

#  training model
model = models.UPENN_GBM_Model_Scratch(models.UPENN_GBM_ViT(), lr=1e-5)

log_dict = model.train(nn.CrossEntropyLoss(), epochs=250, batch_size=468, 
    training_set=training_data, validation_set=validation_data)

train_loss = np.asarray(log_dict['training_loss_per_batch'])
val_loss = np.asarray(log_dict['validation_loss_per_batch'])
train_acc = np.asarray(log_dict['training_acc_per_batch'])
val_acc = np.asarray(log_dict['validation_acc_per_batch'])
num_batches = train_loss.shape[0]

plot_data.plot_line(np.arange(num_batches), train_loss, "train loss", 'Training loss per batch', 'batch number', 'training loss', save=True, fname="train_loss")
plot_data.plot_line(np.arange(num_batches), val_loss, "validation loss", 'validation loss per batch', 'batch number', 'validation loss', save=True, fname="val_loss")

plot_data.plot_line(np.arange(num_batches), train_acc, "train acc", 'Training accuracy per batch', 'batch number', 'training accuracy', save=True, fname="train_acc")
plot_data.plot_line(np.arange(num_batches), val_acc, "validation acc", 'validation accuracy per batch', 'batch number', 'validation accuracy', save=True, fname="val_acc")

