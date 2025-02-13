###############################
# Imports # Imports # Imports #
###############################

import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from typing import Any, Literal
import torchvision.datasets as tvd
import torchvision.transforms.v2 as v2
import mypkg.src.global_config as global_config

# Load config
config = global_config.config

#######################################################
# Models # Models # Models # Models # Models # Models #
#######################################################

def load_stl10(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load STL10 dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.STL10(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor()
        #v2.Resize((413, 775))
    ])

    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 96*96) # Convert to (size, 96*96) from (size, 96,96)

    labels = [d[1] for d in data]
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 10), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_semeion(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load SEMEION dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.SEMEION(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor()
        #v2.Resize((413, 775))
    ])

    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 16*16) # Convert to (size, 16*16) from (size, 16,16)

    labels = [d[1] for d in data]
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 10), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_pcam(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load PCAM dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.PCAM(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor()
        #v2.Resize((413, 775))
    ])

    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 96*96) # Convert to (size, 96*96) from (size, 96,96)

    labels = [d[1] for d in data]
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 2), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_omniglot(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load Omniglot dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.Omniglot(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor()
        #v2.Resize((413, 775))
    ])

    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 105*105) # Convert to (size, 105*105) from (size, 105,105)

    labels = [d[1] for d in data]
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 964), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_fgvcaircraft(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load FGVCAircraft dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.FGVCAircraft(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor(),
        v2.Resize((413, 775))
    ])

    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 413*775) # Convert to (size, 413*775) from (size, 413,775)

    labels = [d[1] for d in data]
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 100), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_dtd(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load DTD dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.DTD(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor(),
        v2.Resize((231, 300))
    ])

    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 231*300) # Convert to (size, 231*300) from (size, 231,300)

    labels = [d[1] for d in data]
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 47), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_eurosat(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load EuroSAT dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.EuroSAT(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor()
    ])

    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 64*64) # Convert to (size, 32*32) from (size, 32,32)

    labels = [d[1] for d in data]
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 10), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_yalefaces(classes:list[int,int]=None, dataset_size:int=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    YaleFaces dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    X = [] # Store images
    labels = [] # Store labels
    
    # Extract images as a vector of (243, 320)
    # Extract labels as an integer from 01 to 15
    filelist = os.listdir(os.path.join(config.yalefaces_dir))
    for fname in filelist:
        label = ''.join([s for s in fname if s.isdigit()])
        label = int(label) - 1
        fname = os.path.join(config.yalefaces_dir, fname)
        img = np.array(Image.open(fname))
        img = img.reshape(img.shape[0]*img.shape[1])
        X.append(img)
        labels.append(label)

    # Stack X and convert X and labels to torch
    X = np.stack(X, axis=0)
    X = torch.from_numpy(X)
    X = X.float().div_(255.)
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 15), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_cifar10(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load CIFAR10 dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5]) (NOT USED)  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.CIFAR10(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor()
    ])

    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 32*32) # Convert to (size, 32*32) from (size, 32,32)

    labels = [d[1] for d in data]
    labels = torch.tensor(labels, dtype=torch.int32)

    # Turn Y into a one-hot encoding
    Y = torch.zeros((X.shape[0], 10), dtype=X.dtype, device=X.device)
    Y[list(range(X.shape[0])), labels] = 1.

    # Extract specific class or downsize dataset
    if classes is not None: 
        # Select only the two classes that are in classes
        mask = (labels.eq(classes[0]) + labels.eq(classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def load_celeba(dataset_size:int = None, classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load CelebA dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5]) (NOT USED)  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    data = tvd.CelebA(config.dataset_dir, download=True)
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToTensor()
    ])
    X = [transforms(d[0])[0] for d in data]
    X = torch.stack(X)
    X = X.view(X.shape[0], 218*178) # Convert to (size, 218*178) from (size, 218, 178)
    if dataset_size is not None: 
        # Downsize dataset to config.dataset_size
        X = X[0:dataset_size]
    return X, None

def load_mnist(dataset:Literal["MNIST", "FMNIST"], dataset_size:int = None, mnist_classes:list[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load (F)MNIST dataset. Optionally downsize dataset size and select subset of two classes.

    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  

    Returns: (`X`, `Y`) data and labels respectively
    """
    if dataset == 'FMNIST':
        mnist = tvd.FashionMNIST(config.dataset_dir, download=True)
    elif dataset == 'MNIST':
        mnist = tvd.MNIST(config.dataset_dir, download=True)
    X = mnist.data.float().div_(255.) # Convert to [0,1] from [0,255]
    X = X.view(X.shape[0], 784) # Convert to (size, 784) from (size, 28, 28)
    Y = torch.zeros((X.shape[0], 10), dtype=X.dtype, device=X.device) # Create labels
    Y[list(range(X.shape[0])), mnist.targets] = 1. # Populate labels
    if mnist_classes is not None: 
        # Select only the two classes that are in config.mnist_classes
        mask = (mnist.targets.eq(mnist_classes[0]) + mnist.targets.eq(mnist_classes[1])).gt(0)
        X = X[mask]
        Y = Y[mask]
    if dataset_size is not None: 
        # Downsize dataset to config.dataset_size
        X, Y = X[0:dataset_size], Y[0:dataset_size]
    return X, Y

def ift(k, x, A=None, phi=None, normalize=True):
    """
    TODO
    """
    if phi is None: 
        phi = torch.zeros(k.shape[0], dtype=k.dtype, device=k.device)
    if A is None: 
        A = torch.ones(k.shape[0], dtype=k.dtype, device=k.device)
    if normalize: 
        A = A/A.sum()
    _2PI = np.pi * 2
    return (torch.sin_(_2PI * k @ x.t() + phi[:, None]) * A[:, None]).sum(0)

def binarize_dataset(X: torch.Tensor, Y: torch.Tensor, classes: list[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a loaded dataset using a 2-class subset, convert the labels to be either -1 or 1.

    `X`: Dataset samples  
    `Y`: Dataset labels  
    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  

    Returns: Binarized dataset (`X`, `Y`)
    """
    if classes is not None:
        Y[:, classes[0]] *= -1
        Y = Y.sum(-1)
    return X, Y

def get_dataset(dataset: Literal['MNIST', 'FMNIST', 'YaleFaces', 'CelebA',  'CIFAR10', 'DTD', 'EuroSAT', 'FGVCAircraft', 'Omniglot', 'PCAM', 'SEMEION', 'STL10'], dataset_size: int = None, classes: list[int, int]= None, val: bool = False, val_split: float = False) -> tuple[torch.Tensor, torch.Tensor, torch.Size]: 
    """
    Get a dataset, binarize if necessary, add noise if necessary.

    `dataset`: String of dataset to load  
    `classes`: Optional, list of two classes to keep in the dataset (ie. [3,5])  
    `dataset_size`: Optional, truncate dataset to be of a specific size  
    `val`: Get validation split  
    `val_split`: Proportion of data for validation split  
    `noise_train`: Add training noise?  

    Returns: (`X`, `Y`, `shape`): training data and labels
    """
    if dataset in ["MNIST", "FMNIST"]:
        X, _ = load_mnist(dataset, dataset_size, classes) # Load dataset
        shape = torch.Size([28,28]) # Image original shape
    elif dataset in ["YaleFaces"]:
        X, _ = load_yalefaces(dataset_size, classes) # Load dataset
        shape = torch.Size([243,320]) # Image original shape
    elif dataset in ["CelebA"]:
        X, _ = load_celeba(dataset_size, classes) # Load dataset
        shape = torch.Size([218, 178]) # Image original shape
    elif dataset in ["CIFAR10"]:
        X, _ = load_cifar10(dataset_size, classes) # Load dataset
        shape = torch.Size([32, 32]) # Image original shape
    elif dataset in ["DTD"]:
        X, _ = load_dtd(dataset_size, classes) # Load dataset
        shape = torch.Size([231, 300]) # Image original shape
    elif dataset in ["EuroSAT"]:
        X, _ = load_eurosat(dataset_size, classes) # Load dataset
        shape = torch.Size([64, 64]) # Image original shape
    elif dataset in ["FGVCAircraft"]:
        X, _ = load_fgvcaircraft(dataset_size, classes) # Load dataset
        shape = torch.Size([413, 775]) # Image original shape
    elif dataset in ["Omniglot"]:
        X, _ = load_omniglot(dataset_size, classes) # Load dataset
        shape = torch.Size([105, 105]) # Image original shape
    elif dataset in ["PCAM"]:
        X, _ = load_pcam(dataset_size, classes) # Load dataset
        shape = torch.Size([96, 96]) # Image original shape
    elif dataset in ["SEMEION"]:
        X, _ = load_semeion(dataset_size, classes) # Load dataset
        shape = torch.Size([16, 16]) # Image original shape
    elif dataset in ["STL10"]:
        X, _ = load_stl10(dataset_size, classes) # Load dataset
        shape = torch.Size([96, 96]) # Image original shape
    else:
        raise Exception(f"Dataset {dataset} does not exist.")

    # X: shape(# data, len*width)
    # Y: shape(# data, # classes) or None
    # X, Y = binarize_dataset(X, Y, classes) # If classes is not None

    # Split depending on if it's a validation subset or not
    if not val: 
        stop = int((1.-val_split) * X.shape[0])
        X = X[:stop]
        #Y = Y[:stop]
    else:
        start = int((1.-val_split) * X.shape[0])
        X = X[start:]
        #Y = Y[start:]
    return X, None, shape
