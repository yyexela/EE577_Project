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
class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
            act_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1), 
            act_fn,
            nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (16, 16)
            act_fn,
            nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (8, 8)
            act_fn,
            nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear(4*out_channels*8*8, latent_dim),
            act_fn
        )

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        output = self.net(x)
        return output


#  defining decoder
class Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()):
        super().__init__()

        self.out_channels = out_channels

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4*out_channels*8*8),
            act_fn
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (8, 8)
            act_fn,
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, 
                                stride=2, output_padding=1), # (16, 16)
            act_fn,
            nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, 
                                stride=2, output_padding=1), # (32, 32)
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
        )

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, 4*self.out_channels, 8, 8)
        output = self.conv(output)
        return output

#  defining autoencoder
class Autoencoder(nn.Module):
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
    def __init__(self, autoencoder):
        self.network = autoencoder
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

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
                images = images.to(config.device)
                #  reconstructing images
                output = self.network(images)
                #  computing loss
                loss = loss_function(output, images.view(-1, 3, 32, 32))
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
                    val_images = val_images.to(config.device)
                    #  reconstructing images
                    output = self.network(val_images)
                    #  computing validation loss
                    val_loss = loss_function(output, val_images.view(-1, 3, 32, 32))

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
                test_images = test_images.to(config.device)
                with torch.no_grad():
                    #  reconstructing test images
                    reconstructed_imgs = self.network(test_images)
                #  sending reconstructed and images to cpu to allow for visualization
                reconstructed_imgs = reconstructed_imgs.cpu()
                test_images = test_images.cpu()

                #  visualisation
                imgs = torch.stack([test_images.view(-1, 3, 32, 32), reconstructed_imgs], 
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

class AE_static(nn.Module):
    """
    Creates a simple linear MLP AutoEncoder.

    `in_dim`: input and output dimension   
    `bottleneck_dim`: dimension at bottleneck  
    `width`: width of model   
    `device`: which device to use   
    """
    def __init__(self, in_dim: int, sizes: list[int], device: str = 'cpu'):
        super(AE_static, self).__init__()
        # Class variables
        self.in_dim = in_dim
        self.sizes = sizes
        self.out_dim = in_dim
        self.device = device

        # Model layer sizes
        print("Creating model with layers:")
        print(sizes)
        print()

        # Define model layers
        self.layers = []
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[idx], sizes[idx+1]))
            if idx != (len(sizes)-2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out

class AE(nn.Module):
    """
    Creates a simple linear MLP AutoEncoder.

    `in_dim`: input and output dimension   
    `bottleneck_dim`: dimension at bottleneck  
    `width`: width of model   
    `device`: which device to use   
    """
    def __init__(self, in_dim: int, bottleneck_dim: int, depth: int, device: str = 'cpu'):
        super(AE, self).__init__()
        # Class variables
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.depth = depth
        self.out_dim = in_dim
        self.device = device

        # Model layer sizes
        sizes = list()
        if depth == 2:
            sizes.extend([in_dim, in_dim])
        elif depth % 2 == 0:
            sizes.extend(np.linspace(in_dim, bottleneck_dim, depth//2, dtype=int).tolist())
            sizes.extend(np.linspace(bottleneck_dim, in_dim, depth//2, dtype=int).tolist())
        else:
            sizes.extend(np.linspace(in_dim, bottleneck_dim, depth//2+1, dtype=int).tolist())
            sizes.extend(np.linspace(bottleneck_dim, in_dim, depth//2+1, dtype=int).tolist()[1:])


        print("Creating model with layers:")
        print(sizes)
        print()

        # Define model layers
        self.layers = []
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[idx], sizes[idx+1]))
            if idx != (len(sizes)-2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


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
