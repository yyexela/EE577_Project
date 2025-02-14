###############################
# Imports # Imports # Imports #
###############################

import os
import torch
import scipy
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm as tqdm_regular
from typing import Any
from functools import reduce
from argparse import Namespace
from scipy.optimize import curve_fit
from src import models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error
import src.global_config as global_config

# Load config
config = global_config.config

#########################################
# Generic Functions # Generic Functions #
#########################################

def min_max_fit(A: np.ndarray, a:float=0., b:float=1.) -> np.ndarray:
    """
    Scale the array A to have values in the range [min,max] globally

    `A`: Input array  
    `a`: Minimum value  
    `b`: Maximum value  

    Returns: Scaled array `A`  
    """

    denom = A.max() - A.min()
    if np.abs(denom) <= 1e-8:
        raise Exception("min_max_fit: denominator is 0")
    A_scaled = a + (A - A.min())*(b-a)/(denom)
    
    return A_scaled, A.min(), A.max()

def min_max_fit_inv(A_scaled: np.ndarray,min_A: np.ndarray, max_A: np.ndarray, a:float=0., b:float=1.) -> np.ndarray:
    """
    Scale the array A to have values in the range [min,max] globally

    `A`: Input array  
    `a`: Original minimum value  
    `b`: Original maximum value  
    `min_A`: Input array min  
    `min_A`: Input array max  

    Returns: Scaled array `A`  
    """

    A = (A_scaled*(max_A - min_A) - a)/(b-a) + min_A
    
    return A

def interpolate(input_u: np.ndarray, input_x: np.ndarray, output_x: np.ndarray, axis:int = 0) -> np.ndarray:
    """
    Given a 2D array of values `input_u` and domain `input_x`, interpolate the input along `axis` over a new domain `output_x`. Assumes 0 everywhere outside of original domain.
    
    Returns an linear interpolation of `input_u` over domain `output_x`
    """
    if axis == 1:
        input_u = input_u.T

    # Set up variables
    length = input_u.shape[0] # Number of times to run interpolation
    dx = output_x[1] - output_x[0] # dx for output x
    x_tmp = np.arange(input_x[0],input_x[-1]+dx,dx) # input domain at dx resolution

    # Interpolate
    u_interp = np.zeros((length, output_x.shape[0])) # output to be filled
    for i in range(length):
        u_tmp = np.interp(x_tmp, input_x, input_u[i,:])
        x1_idx = np.where(output_x>=x_tmp[0])[0][0]
        x2_idx = x1_idx + u_tmp.shape[0]
        u_interp[i,x1_idx:x2_idx] = u_tmp

    if axis == 1:
        u_interp = u_interp.T

    return u_interp

def fit_gaussians(u: np.ndarray, x: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a 2D array of values `u`, and domain `x`, fit gaussians along specified axis (ie. dimension) and return the means, stdevs, and gaussians
    """
    means = list()
    stdevs = list()
    amps = list()
    gaussians = list()

    # Fit gaussians
    for idx in range(u.shape[axis]):
        ux = u[idx,:] if axis == 0 else u[:,idx]
        popt, pcov = curve_fit(gaussian, x, ux)
        mu = popt[0]
        sigma = popt[1]
        amplitude = popt[2]

        means.append(mu)
        stdevs.append(sigma)
        amps.append(amplitude)
        fit = gaussian(x, mu, sigma, amplitude)
        gaussians.append(fit)

    means = np.asarray(means)
    stdevs = np.asarray(stdevs)
    amps = np.asarray(amps)
    gaussians = np.stack(gaussians)

    if axis == 1:
        gaussians = gaussians.T

    return means, stdevs, amps, gaussians

def moving_average(a: np.ndarray, n: int=1, axis: int=None) -> np.ndarray:
    """
    Smooth input matrix

    `a`: Input matrix to smooth  
    `n`: Number of elements to compute average for for smoothing  
    `axis`: Axis to smooth over  

    Returns: Smoothed matrix  
    """
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def print_dictionary(d: dict[str, str]) -> None:
    """
    Print given dictionary with keys and values

    `d`: Hyperparameters dictionary to print key and values for

    Returns: `None`
    """
    for key in d.keys():
        print(f"{key}: {d[key]}")

    return None

def gaussian(x: np.ndarray, mu: float, sigma:float, a: float) -> np.ndarray:
    """
    Simple 1D Gaussian Function

    `x`: Input values to evaluate gaussian at  
    `mu`: Mean of gaussian  
    `sigma`: Standard deviation of gaussian  
    `a`: Amplitude  

    Returns:  
    Values of gaussian evaluated at `x`  
    """
    return (a * 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.0) / 2))

def fit_gaussian(x: np.ndarray, y: np.ndarray, p0:list[Any]=None) -> list[Any]:
    """
    First input data y to a 1D gaussian function


    `x`: Input domain values  
    `y`: Input range value to fit gaussian to   
    `p0`: Initial guess for curve fit  

    Returns: (`mu`, `sigma`, `a`)  
    Mean, standard deviation, and amplitude of Gaussian
    """
    return scipy.optimize.curve_fit(gaussian, x, y, p0)[0]

def get_AE_output(model, dataset, dataset_size, classes, device='cpu'):
    """
    Given a trained AE, generate its output

    """
    # Build loss and keep track of it
    loss_fn = nn.MSELoss()
    loss_cum = 0; loss_count = 0;
    # Load dataset
    input_, _, _ = datasets.get_dataset(dataset, dataset_size, classes, val=False)
    # Collect AE output
    outputs = list()
    dl = DataLoader(input_, batch_size=1024, shuffle=False)
    with torch.no_grad(): 
        for i, data in enumerate(dl):
            data = data.to(device)
            x = data
            yt = data.view(-1, model.out_dim)
            y = model(x).detach().clone()
            loss_cum += loss_fn(y, yt).cpu().item()
            loss_count += 1
            outputs.append(y.cpu())
    outputs = torch.cat(outputs)
    outputs = outputs.cpu().numpy()

    print("AE average loss:", loss_cum/loss_count)

    return outputs

def train_AE_static(dataset, dataset_size, classes, lr, iters, mlp_in_dim, sizes, device):
    """
    Train AutoEncoder


    """
    tX, _, _ = datasets.get_dataset(dataset, dataset_size, classes, val=False)
    model = models.AE_static(mlp_in_dim,\
                        sizes,\
                        device)
    model = train_AE_model(model, lr, iters, tX, device)
    output_ = get_AE_output(model, dataset, dataset_size, classes, device)
    return output_

def layer_counter(layers):
    """
    Count number of parameters in an MLP with the provided layers. (ex. [784, 361, 784])
    """
    count = 0
    for i in range(len(layers)):
        count += layers[i] * layers[i+1]
        if i == len(layers)-2:
            break

    return count

def create_p_norm_full(shape, p):
    """
    Create a matrix showing the distances of the specified p-norm from
    the center of the shape.

    `shape`: Shape of the mask (height, width)  
    `radius`: Radius of the norm  
    `p`: The "p" in "p-norm"  

    Returns: `mask`  
    Distances of each point in the original image size from the origin according
    to the provided p-norm.

    Note: To do scaled mask just multiply xx or yy by some scaling factor
    """
    # Calculate the center of the matrix
    center_col, center_row = shape[0] / 2. - 0.5, shape[1] / 2. - 0.5

    # Calculate distances
    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    yy, xx = np.meshgrid(x, y, indexing='xy')
    xx = np.abs(center_row - xx.T)
    yy = np.abs(center_col - yy.T)
    stacked = np.stack([xx,yy])
    dist = np.linalg.norm(stacked, ord=p, axis=0)

    return dist

def create_p_norm_mask(shape, p, radius, smoothing: int = 0.5):
    """
    Create a mask of the specified p-norm with the radius corresponding to
    distance of the y=x intersection in the first quadrant from the center
    of the norm.

    `shape`: Shape of the mask (height, width)  
    `radius`: Radius of the norm  
    `p`: The "p" in "p-norm"  
    `smoothing`: distance from radius to include in mask (default is 0.5,
                 larger means more values) (ie, range for mask from distance
                 is [radius-smoothing, radius+smoothing])  

    Returns: `mask`  
    Binary mask with values of 1 on the norm and 0 outside.
    """
    # Calculate distances from center of the shape
    dist = create_p_norm_full(shape, p)

    # Get mask outline of shape for specified radius
    mask1 = (dist >= radius-smoothing).astype(int)
    mask2 = (dist <= radius+smoothing).astype(int)
    mask = mask1*mask2

    return mask

def to_torch_dataset_1d(t: np.ndarray, yt: np.ndarray, device='cpu') -> tuple[torch.tensor, torch.tensor]:
    """
    Given x and y values, create a pytorch dataset for training

    `t`: `x-values`  
    `yt`: `y-values`  

    Returns: (`t`, `yt`) where `t` is a torch tensor of x-values and `yt` is a torch tensor of y-values.
    """
    t = torch.from_numpy(t).view(-1, 1).float()
    yt = torch.from_numpy(yt).view(-1, 1).float()
    t = t.to(device)
    yt = yt.to(device)
    return t, yt

def get_MLP_hyperparameters(mlp_option, mlp_in_dim, dataset):
    if dataset in ['MNIST', 'FMNIST']:
        # Depth of 3 or 5
        # width from: mlp_in_dim / 1 to 28 squared
        mlp_bottleneck_dims = np.asarray([i ** 2 for i in range(1,29)])
        mlp_depths = [3,5,6,4]
        
        mlp_bottleneck_dim = mlp_bottleneck_dims[mlp_option % 28]
        mlp_depth = mlp_depths[mlp_option // 28]

        if mlp_option >= 28*4 :
            raise Exception("Invalid mlp_option")
    elif dataset in ['Omniglot', 'STL10', 'SEMEION', 'PCAM']:
        match mlp_option:
            case 0:
                mlp_bottleneck_dim = mlp_in_dim//32
                mlp_depth = 5
            case 1:
                mlp_bottleneck_dim = mlp_in_dim//16
                mlp_depth = 5
            case 2:
                mlp_bottleneck_dim = mlp_in_dim//8
                mlp_depth = 5
            case 3:
                mlp_bottleneck_dim = mlp_in_dim//32
                mlp_depth = 3
            case 4:
                mlp_bottleneck_dim = mlp_in_dim//16
                mlp_depth = 3
            case 5:
                mlp_bottleneck_dim = mlp_in_dim//8
                mlp_depth = 3
            case _:
                raise Exception("Invalid mlp_option")
    elif dataset in ['EuroSAT', 'YaleFaces', 'CelebA']:
        match mlp_option:
            case 0:
                mlp_bottleneck_dim = int(np.sqrt(mlp_in_dim))*8
                mlp_depth = 5
            case 1:
                mlp_bottleneck_dim = int(np.sqrt(mlp_in_dim))*4
                mlp_depth = 5
            case 2:
                mlp_bottleneck_dim = int(np.sqrt(mlp_in_dim))*2
                mlp_depth = 5
            case 3:
                mlp_bottleneck_dim = int(np.sqrt(mlp_in_dim))*8
                mlp_depth = 3
            case 4:
                mlp_bottleneck_dim = int(np.sqrt(mlp_in_dim))*4
                mlp_depth = 3
            case 5:
                mlp_bottleneck_dim = int(np.sqrt(mlp_in_dim))*2
                mlp_depth = 3
            case _:
                raise Exception("Invalid mlp_option")
    else:
        raise Exception("Invalid dataset")
    
    return mlp_bottleneck_dim, mlp_depth

def extract_each_class(dataset):
    """
    This function searches for and returns
    one image per class
    """
    images = []
    ITERATE = True
    i = 0
    j = 0

    while ITERATE:
        for label in tqdm_regular(dataset.targets):
            if label==j:
                images.append(dataset.data[i])
                print(f'class {j} found')
                i+=1
                j+=1
            if j==10:
                ITERATE = False
        else:
            i+=1

    return images

