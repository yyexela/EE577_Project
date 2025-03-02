###############################
# Imports # Imports # Imports #
###############################

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import src.global_config as global_config
import os

# Load config
config = global_config.config

#######################################################
# Models # Models # Models # Models # Models # Models #
#######################################################

#  defining encoder
class CIFAR10Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=200, img_len=32, act_fn=nn.ReLU()):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_len = img_len
        self.out_img_len = 8

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.in_channels, padding=1), # (32, 32)
            act_fn,
            nn.Conv2d(out_channels, out_channels, self.in_channels, padding=1), 
            act_fn,
            nn.Conv2d(out_channels, 2*out_channels, self.in_channels, padding=1, stride=2), # (16, 16)
            act_fn,
            nn.Conv2d(2*out_channels, 2*out_channels, self.in_channels, padding=1),
            act_fn,
            nn.Conv2d(2*out_channels, 4*out_channels, self.in_channels, padding=1, stride=2), # (8, 8)
            act_fn,
            nn.Conv2d(4*out_channels, 4*out_channels, self.in_channels, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear(4*out_channels*int(self.out_img_len**2), latent_dim),
            act_fn
        )

        print("Encoder:")
        print(self.net)
        print()

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.img_len, self.img_len)
        output = self.net(x.float())
        return output


#  defining decoder
class CIFAR10Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=200, img_len=32, act_fn=nn.ReLU()):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_len = img_len
        self.out_img_len = 8

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4*out_channels*int(self.out_img_len**2)),
            act_fn
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(4*out_channels, 4*out_channels, self.in_channels, padding=1), # (8, 8)
            act_fn,
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, self.in_channels, padding=1, 
                                stride=2, output_padding=1), # (16, 16)
            act_fn,
            nn.ConvTranspose2d(2*out_channels, 2*out_channels, self.in_channels, padding=1),
            act_fn,
            nn.ConvTranspose2d(2*out_channels, out_channels, self.in_channels, padding=1, 
                                stride=2, output_padding=1), # (32, 32)
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, self.in_channels, padding=1),
            act_fn,
            nn.ConvTranspose2d(out_channels, in_channels, self.in_channels, padding=1)
        )

        print("Decoder:")
        print(self.linear)
        print(self.conv)
        print()

    def forward(self, x):
        output = self.linear(x.float())
        output = output.view(-1, 4*self.out_channels, self.out_img_len, self.out_img_len)
        output = self.conv(output)
        return output

#  defining autoencoder
class CIFAR10Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(config.device)

        self.decoder = decoder
        self.decoder.to(config.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class UPENNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=200, img_len=32, act_fn=nn.ReLU()):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_len = img_len
        self.out_img_len = 27

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.in_channels, padding=1), # (32, 32)
            act_fn,
            nn.Conv2d(out_channels, out_channels, self.in_channels, padding=1), 
            act_fn,
            nn.Conv2d(out_channels, 2*out_channels, self.in_channels, padding=1, stride=2), # (16, 16)
            act_fn,
            nn.Conv2d(2*out_channels, 2*out_channels, self.in_channels, padding=1),
            act_fn,
            nn.Conv2d(2*out_channels, 4*out_channels, self.in_channels, padding=1, stride=2), # (8, 8)
            act_fn,
            nn.Conv2d(4*out_channels, 4*out_channels, self.in_channels, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear(4*out_channels*int(self.out_img_len**2), latent_dim*4),
            act_fn,
            nn.Linear(latent_dim*4, latent_dim*2),
            act_fn,
            nn.Linear(latent_dim*2, latent_dim),
            act_fn
        )

        print("Encoder:")
        print(self.net)
        print()

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.img_len, self.img_len)
        output = self.net(x.float())
        return output


#  defining decoder
class UPENNDecoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=200, img_len=32, act_fn=nn.ReLU()):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_len = img_len
        self.out_img_len = 27

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            act_fn,
            nn.Linear(latent_dim*2, latent_dim*4),
            act_fn,
            nn.Linear(latent_dim*4, 4*out_channels*int(self.out_img_len**2)),
            act_fn
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(4*out_channels, 4*out_channels, self.in_channels, padding=1), # (8, 8)
            act_fn,
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, self.in_channels, padding=1, 
                                stride=2, output_padding=1), # (16, 16)
            act_fn,
            nn.ConvTranspose2d(2*out_channels, 2*out_channels, self.in_channels, padding=1),
            act_fn,
            nn.ConvTranspose2d(2*out_channels, out_channels, self.in_channels, padding=1, 
                                stride=2, output_padding=1), # (32, 32)
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, self.in_channels, padding=1),
            act_fn,
            nn.ConvTranspose2d(out_channels, in_channels, self.in_channels, padding=1)
        )

        print("Decoder:")
        print(self.linear)
        print(self.conv)
        print()

    def forward(self, x):
        output = self.linear(x.float())
        output = output.view(-1, 4*self.out_channels, self.out_img_len, self.out_img_len)
        output = self.conv(output)
        return output

#  defining autoencoder
class UPENNAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(config.device)

        self.decoder = decoder
        self.decoder.to(config.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ConvolutionalAutoencoder():
    def __init__(self, autoencoder, in_channels = 3, img_len = 32):
        self.network = autoencoder
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.in_channels = in_channels
        self.img_len = img_len

    def train(self, loss_function, epochs, batch_size, 
            training_set, validation_set, test_set):
    
        #  creating log
        log_dict = {
            'training_loss_per_batch': [],
            'validation_loss_per_batch': [],
            'visualizations': []
        } 

        #  defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)

        #  initializing network weights
        self.network.apply(init_weights)

        #  creating dataloaders
        train_loader = DataLoader(training_set, batch_size)
        val_loader = DataLoader(validation_set, batch_size)
        test_loader = DataLoader(test_set, 10)

        #  setting convnet to training mode
        self.network.train()
        self.network.to(config.device)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            train_losses = []

            #------------
            #  TRAINING
            #------------
            print('training...')
            for images in tqdm(train_loader):
                #  zeroing gradients
                self.optimizer.zero_grad()
                #  sending images to device
                images = images.to(config.device).float()
                #  reconstructing images
                output = self.network(images)
                #  computing loss
                loss = loss_function(output, images.view(-1, self.in_channels, self.img_len, self.img_len))
                #  calculating gradients
                loss.backward()
                #  optimizing weights
                self.optimizer.step()

                #--------------
                # LOGGING
                #--------------
                log_dict['training_loss_per_batch'].append(loss.item())

            #--------------
            # VALIDATION
            #--------------
            print('validating...')
            for val_images in tqdm(val_loader):
                with torch.no_grad():
                    #  sending validation images to device
                    val_images = val_images.to(config.device).float()
                    #  reconstructing images
                    output = self.network(val_images)
                    #  computing validation loss
                    val_loss = loss_function(output, val_images.view(-1, self.in_channels, self.img_len, self.img_len))

                #--------------
                # LOGGING
                #--------------
                log_dict['validation_loss_per_batch'].append(val_loss.item())


            #--------------
            # VISUALISATION
            #--------------
            print(f'training_loss: {round(loss.item(), 4)} validation_loss: {round(val_loss.item(), 4)}')

            for test_images in test_loader:
                #  sending test images to device
                test_images = test_images.to(config.device).float()
                with torch.no_grad():
                    #  reconstructing test images
                    reconstructed_imgs = self.network(test_images)
                #  sending reconstructed and images to cpu to allow for visualization
                reconstructed_imgs = reconstructed_imgs.cpu()
                test_images = test_images.cpu()

                #  visualisation
                imgs = torch.stack([test_images.view(-1, self.in_channels, self.img_len, self.img_len), reconstructed_imgs], 
                                    dim=1).flatten(0,1)
                grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
                grid = grid.permute(1, 2, 0)
                plt.figure(dpi=170)
                plt.title('Original/Reconstructed')
                plt.imshow(grid)
                log_dict['visualizations'].append(grid)
                plt.axis('off')
                plt.show()
                plt.savefig(os.path.join(config.image_dir, f'Visualizations_{epoch}.pdf'))
                plt.close()
            
        return log_dict

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        encoder = self.network.encoder
        return encoder(x)
    
    def decode(self, x):
        decoder = self.network.decoder
        return decoder(x)

class UPENN_GBM_MLPs(nn.Module):
    def __init__(self, in_channels = 768, n_modalities = 4, out_bins = 7):
        super().__init__()

        self.n_modalities = n_modalities

        self.modality_mlps = [
            MLP(in_channels, in_channels//2, 3, in_channels//4, device = config.device)
        ] * self.n_modalities
        self.combined_mlp = MLP(in_channels, in_channels//4, 3, out_bins, device = config.device)

        for mlp in self.modality_mlps:
            mlp.to(config.device)
        self.combined_mlp.to(config.device)

    def forward(self, x):
        x_outs = list()
        for modality_idx in range(self.n_modalities):
            x_outs.append(self.modality_mlps[modality_idx](x[:,modality_idx,:]).unsqueeze(1))
        x_outs = torch.cat(x_outs,dim=1)
        x_outs = x_outs.reshape(x.shape[0],-1)
        output = self.combined_mlp(x_outs)
        return output

class UPENN_GBM_Model():
    def __init__(self, MLP_model, in_channels = 768, n_modalities = 4, lr=1e-4):
        self.network = MLP_model
        self.n_modalities = n_modalities
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.in_channels = in_channels

    def train(self, loss_function, epochs, batch_size, 
            training_set, validation_set):
    
        #  creating log
        log_dict = {
            'training_loss_per_batch': [],
            'validation_loss_per_batch': [],
            'visualizations': []
        } 

        #  defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)

        #  initializing network weights
        self.network.apply(init_weights)

        #  creating dataloaders
        train_loader = DataLoader(training_set, batch_size)
        val_loader = DataLoader(validation_set, batch_size)

        #  setting convnet to training mode
        self.network.train()
        self.network.to(config.device)

        for epoch in tqdm(range(epochs)):
            train_losses = []

            #------------
            #  TRAINING
            #------------
            for _, (images, labels) in enumerate(train_loader):
                #  zeroing gradients
                self.optimizer.zero_grad()
                #  sending images to device
                images = images.to(config.device).float()
                labels = labels.to(config.device).float()
                #  reconstructing images
                output = self.network(images)
                #  computing loss
                loss = loss_function(output, labels)
                #  calculating gradients
                loss.backward()
                #  optimizing weights
                self.optimizer.step()

                #--------------
                # LOGGING
                #--------------
                log_dict['training_loss_per_batch'].append(loss.item())

            #--------------
            # VALIDATION
            #--------------
            for _, (val_images, val_labels) in enumerate(val_loader):
                with torch.no_grad():
                    #  sending validation images to device
                    val_images = val_images.to(config.device).float()
                    val_labels = val_labels.to(config.device).float()
                    #  reconstructing images
                    val_output = self.network(val_images)
                    #  computing validation loss
                    val_loss = loss_function(val_output, val_labels)

                #--------------
                # LOGGING
                #--------------
                log_dict['validation_loss_per_batch'].append(val_loss.item())

        return log_dict

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        encoder = self.network.encoder
        return encoder(x)
    
    def decode(self, x):
        decoder = self.network.decoder
        return decoder(x)

class MLP(nn.Module):
    """
    Creates a simple linear MLP.

    `in_dim`: input dimension   
    `width`: width of model   
    `depth`: depth of model   
    `out_dim`: output dimension   
    `device`: which device to use   
    """
    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int, device: str = 'cpu'):
        super(MLP, self).__init__()
        # Class variables
        self.in_dim = in_dim
        self.width = width
        self.depth = depth
        self.out_dim = out_dim
        self.device = device

        # Define model layers
        self.layers = []
        self.layers.append(nn.Linear(in_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(depth - 2): 
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.extend([nn.Linear(width, out_dim)])

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out
