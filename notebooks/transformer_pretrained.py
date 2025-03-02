
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

# Load dataset
embedding_dict = helpers.get_pretrained_embeddings(re_run = False)
embedding_tensor = torch.cat([embedding_dict[mod].unsqueeze(1) for mod in ['T2', 'FLAIR', 'T1', 'T1GD']], dim=1)
labels = embedding_dict['labels']

# Split it 80 - 20
split_idx = int(embedding_tensor.shape[0]*0.8)
training_data = embedding_tensor[0:split_idx,:,:]
training_labels = labels[0:split_idx,:]
validation_data = embedding_tensor[split_idx:,:,:]
validation_labels = labels[split_idx:,:]

# Set up dataset
training_data = datasets.CustomLabeledDataset(training_data, training_labels)

validation_data = datasets.CustomLabeledDataset(validation_data, validation_labels)

#  training model
model = models.UPENN_GBM_Model(models.UPENN_GBM_MLPs(), lr=1e-5)

log_dict = model.train(nn.CrossEntropyLoss(), epochs=10000, batch_size=468, 
    training_set=training_data, validation_set=validation_data)

train_loss = np.asarray(log_dict['training_loss_per_batch'])
val_loss = np.asarray(log_dict['validation_loss_per_batch'])
num_batches = train_loss.shape[0]

plot_data.plot_line(np.arange(num_batches), train_loss, "train loss", 'Training loss per batch', 'batch number', 'training loss', save=True, fname="train_loss")
plot_data.plot_line(np.arange(num_batches), val_loss, "validation loss", 'validation loss per batch', 'batch number', 'validation loss', save=True, fname="val_loss")

