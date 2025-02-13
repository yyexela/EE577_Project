###############################
# Imports # Imports # Imports #
###############################

import numpy as np
import torch.nn as nn
import mypkg.src.global_config as global_config

# Load config
#config = global_config.config

#######################################################
# Models # Models # Models # Models # Models # Models #
#######################################################

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
