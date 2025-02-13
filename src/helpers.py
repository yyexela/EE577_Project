###############################
# Imports # Imports # Imports #
###############################

import os
import torch
import scipy
import pickle
import numpy as np
from torch import nn
import pysindy as ps
from tqdm import tqdm
from typing import Any
from functools import reduce
from argparse import Namespace
from scipy.optimize import curve_fit
from mypkg.src import models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error
import mypkg.src.global_config as global_config

# Load config
config = global_config.config

#########################################
# Generic Functions # Generic Functions #
#########################################

def pysindy_grid_search(pde_lib, u: np.ndarray, x: np.ndarray, t: np.ndarray, train_split: float, lower_thresh: float, upper_thresh: float, alpha: float):
    """
    Do complicated search over threshold and alpha for PySINDY's STLSQ model  

    First, gets the number of terms from upper and lower threshold

    Then, calculates expected number of terms between them (ie, if upper threshold gives 0 and lower threshold gives 6, returns list of [0,1,2,3,4,5,6])

    Then, for each next lower unfound integer number of terms (ie. 1), does bisection search for tolerance between next lowest and next highest number of terms

    Also calculates tolerance for bisection to break an infinite loop

    Also calculates timeout for a specific number to break an infinite loop
    """
    # Convert to PySINDy format
    if u.shape[1] != len(t):
        u = u.T
    if u.shape[1] != len(t):
        u = u.T
        raise Exception(f"u shape is weird: {u.shape}, x shape {x.shape}, t shape {t.shape}")
    u = np.expand_dims(u, -1)
    dt = t[1] - t[0]

    # Split into train and test
    u_train = u[:, :int(train_split*u.shape[1]), :]
    u_test = u[:, int(train_split*u.shape[1]):, :]
    #t_train = t[int(train_split*u.shape[1]):]
    #t_test = t[:int(train_split*u.shape[1])]

    # Keep track of number of terms and their scores
    # score_dict: contains number of terms as keys and sorted list of thresholds as values (sorted in increasing order), alphas, and scores
    #   score_dict[N] = (thresh, alpha, score)
    #   key: N (int)
    #   value: (thresh, alpha, score): tuple(float, float, float)
    score_dict = {}

    # Calculate boundaries
    for threshold in [lower_thresh, upper_thresh]:
        optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True)
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        model.fit(u_train, t=dt)

        #score = model.score(u_test, metric=root_mean_squared_error)
        n_terms = int(np.count_nonzero(model.optimizer.coef_))
        score = model.score(u_test, metric=root_mean_squared_error)

        print("Number of terms:", n_terms)
        print("Score:", score)
        print("Alpha:", alpha)
        print("Threshold:", threshold)
        model.print(precision=5)
        print()

        score_dict[n_terms] = [(threshold, alpha, score)]

    # Numbers of terms we want
    wanted_l = list(range(min(score_dict.keys()), max(score_dict.keys())+1))

    # Helper to identify next threshold
    # Either returns next threshold (> 0) or no more threshold (-1)
    def next_threshold_func(score_dict, wanted_list, tol=1e-8):
        unfound_n = list(set(wanted_list).difference(score_dict.keys()))
        if len(unfound_n) == 0:
            return -1, [], -1
        next_n = sorted(unfound_n)[0]
        lower_thresh = None
        upper_thresh = None
        # Sort in increasing n, but recall that thresholds are decreasing
        for found_n in sorted(list(score_dict.keys()), reverse=True):
            if found_n >= next_n:
                lower_thresh = sorted(score_dict[found_n], key=lambda x: x[0])[-1][0]
            elif found_n <= next_n:
                upper_thresh = sorted(score_dict[found_n], key=lambda x: x[0])[0][0]
                break
        if lower_thresh is None or upper_thresh is None:
            raise Exception(f"lower_thresh ({lower_thresh}) or upper_thresh ({upper_thresh}) is None")
        # If difference between thresholds is within a certain tolerance, skip this number
        if abs(upper_thresh - lower_thresh) <= tol:
            #print(f"Expected n {next_n} not found within tolerance, skipping")
            wanted_list.remove(next_n)
            if wanted_list is None:
                wanted_list = []
        next_thresholds = np.linspace(lower_thresh, upper_thresh, 5)[1:4].tolist()
        return next_n, next_thresholds, wanted_list

    # Grid search different hyperparams
    next_n, next_thresholds, wanted_l = next_threshold_func(score_dict, wanted_l)
    current_term = next_n
    term_counter = 0
    term_timeout = 100
    while(len(next_thresholds) != 0):
        #print("Current terms:", sorted(score_dict.keys()))
        #print("Searching for:", next_n)
        #print("next_thresholds:", next_thresholds)

        for next_threshold in next_thresholds:
            # Using normalize_columns = True to improve performance. (note from a PySINDy notebook)
            optimizer = ps.STLSQ(threshold=next_threshold, alpha=alpha, normalize_columns=True)
            model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
            model.fit(u_train, t=dt)
            n_terms = int(np.count_nonzero(model.optimizer.coef_))
            score = model.score(u_test, metric=root_mean_squared_error)

            #print("n_terms:", n_terms)
                
            current_list = score_dict.get(n_terms, [])
            if len(current_list) == 0:
                pass
                #print("Found new term!", n_terms)
            else:
                term_counter += 1
            current_list.append((next_threshold, alpha, score))
            current_list = sorted(current_list, key=lambda x: x[0])
            score_dict[n_terms] = current_list

        next_n, next_thresholds, wanted_l = next_threshold_func(score_dict, wanted_l)

        if next_n != current_term:
            term_counter = 0
        elif term_counter >= term_timeout:
            #print(f"Timed out for term {current_term}")
            wanted_l = wanted_l.remove(current_term)
            if wanted_l is None:
                wanted_l = []
            term_counter = 0
            next_n, next_thresholds, wanted_l = next_threshold_func(score_dict, wanted_l)
        current_term = next_n

    # Extract largest score from score_dict to get max_dict
    max_dict = {}
    for key in score_dict.keys():
        max_dict[key] = sorted(score_dict[key], key=lambda x: x[2])[-1]

    max_n, min_n = (max(max_dict.keys()), min(max_dict.keys()))
    max_n_threshold, min_n_threshold = (max_dict[max_n], max_dict[min_n])
    #print(f"max_n {max_n} with threshold {max_n_threshold}")
    #print(f"min_n {min_n} with threshold {min_n_threshold}")

    for n_terms in max_dict.keys():
        #print()
        #print(f"n_terms {n_terms}:")
        threshold, alpha, _ = max_dict[n_terms]
        optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True)
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        model.fit(u_train, t=dt)
        model.print(precision=5)

    return max_dict
            

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

def modify_pde_sindy_out(input_str,order=4):
    """
    Convert PySindy PDE text output to be more readable
    """
    output_str = input_str
    for ord in range(order,0,-1):
        output_str = output_str.replace(f"x0_{'1'*ord}", f"u_{'x'*ord}")
    output_str = output_str.replace(f"x0", f"u")
    return output_str

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

def mnist_pde_rhs(t, u0_fft, k, eps, alpha):
    # u0_fft: u0 in frequency domain
    # u0: u0 in space domain
    # u0_x: u0 partial derivative with respect to x

    u0 = np.real(scipy.fft.ifft(u0_fft))

    u0_2_u0_x = 1j*k*scipy.fft.fft(u0**3/3)

    rhs = - eps*(k**2)*u0_fft + alpha*u0_2_u0_x

    return rhs

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

def calculate_integrals(data: np.ndarray, p: float, smoothing: float, radius_mult: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given an image `data`, calculate the radial or diamond integrals

    `data`: Data to calculate integrals over  
    `p`: Integrate over the specified p-norm  
    `smoothing`: distance from radius to include in mask (default is 0.5,
                 larger means more values) (ie, range for mask from distance
                 is [radius-smoothing, radius+smoothing])  
    `radius_mult`: multiplier for radius (useful for p < 1)  

    Returns: (r_x, r_y, ints), all np.ndarray  
    `r_x`: x-values that integrals were calculated over  
    `r_y`: y-values that integrals were calculated voer  
    """
    return_ints = []
    return_x = []
    return_y = []
    rx_vals = list(range(1, int(radius_mult*data.shape[0]//2+1)))
    ry_vals = list()
    for rx in rx_vals:
        ry = max(int(rx*(data.shape[0]/data.shape[1])),1)
        ry_vals.append(ry)
        mask = create_p_norm_mask(shape=data.shape,
                                  p=p,
                                  radius=rx,
                                  smoothing=smoothing)
        if np.sum(mask) != 0:
            int_val = np.sum(data*mask) / np.sum(mask)
            return_ints.append(int_val)
            return_x.append(rx)
            return_y.append(ry)
    
    return (return_x, return_y, return_ints)

def generate_integral_data(svd, shape, fft_pos_freq, fft_normalize, fft_centering, p, smoothing, radius_mult, crop, filename):
    """
    Generate radial integrals across all modes given an SVD of a dataset.

    `svd`: Input SVD of a dataset  
    `shape`: Shape of original image of SVD dataset  
    `fft_pos_freq`: Output positive frequencies  
    `fft_normalize`: Normalize output  
    `fft_centering`: Input signal is centered   
    `p`: The "p" in "p-norm"  
    `smoothing`: distance from radius to include in mask (default is 0.5,
                 larger means more values) (ie, range for mask from distance
                 is [radius-smoothing, radius+smoothing])  
    `radius_mult`: multiplier for radius (useful for p < 1)  
    `crop`: Crop image to middle third  
    `filename`: Filename of saved file if we're saving it  

    Returns: (`d`)  
    `d` is a dictionary containing keys `ints`, `r_x`, and `r_y`. where `ints` are the values of the radial integrals for each mode of length length(d[`r_x`]), where `r_x` and `r_y` are the radii of the ellipse at which the integral was calculated. All elements are numpy arrays.
    """
    pkl_path = os.path.join(config.image_dir, filename)
    int_vals_total = []
    for j in tqdm(range(svd[1].shape[0])):
        zz = svd[2][:,j].reshape(shape[0], shape[1])
        zzf, _ = fft2d(zz, pos_freq=fft_pos_freq, normalize=fft_normalize, centering=fft_centering)
        if crop:
            zzf = zzf[(zzf.shape[0]//3)*1:(zzf.shape[0]//3)*2,(zzf.shape[1]//3)*1:(zzf.shape[1]//3)*2]
        # Calculate integral
        r_x, r_y, int_vals = calculate_integrals(zzf, p, smoothing, radius_mult=radius_mult)
        int_vals_total.append(int_vals)                
    int_vals_total = np.stack(int_vals_total)
    d = {'p': p, 'smoothing': smoothing, 'ints': int_vals_total, 'r_x': r_x, 'r_y': r_y}
    with open(pkl_path, 'wb') as file:
        pickle.dump(d, file)
    return d

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

def train_AE(dataset, dataset_size, classes, lr, iters, mlp_in_dim, mlp_bottleneck_dim, mlp_depth, device):
    """
    Train AutoEncoder


    """
    tX, _, _ = datasets.get_dataset(dataset, dataset_size, classes, val=False)
    model = models.AE(mlp_in_dim,\
                        mlp_bottleneck_dim,\
                        mlp_depth,\
                        device)
    model = train_AE_model(model, lr, iters, tX, device)
    output_ = get_AE_output(model, dataset, dataset_size, classes, device)
    return output_

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

def make_phased_waves_2d(amps: list[int], freqs: list[int], phase_shifts: list[int], n: int):
    """
    Creates 2D overlapping sine waves with specified amplitudes, phase shifts, and frequencies. The domain is [0.,1.]^2 and N^2 points are generated.

    `amps`: List of amplitudes for the sine waves  
    `freqs`: List of frequencies for the sine waves  
    `phase_shifts`: List of phase shifts for the sine waves  
    `n`: Number of points generated for the function on [0.,1.]  

    Returns: (`xx`, `yy`, `zz`)  
    `xx`: x-coordinates  
    `yy`: y-coordinated   
    `zz`: function evaluations  
    """
    t = np.arange(0, 1, 1./n)
    xx, yy = np.meshgrid(t, t)
    zz = reduce(lambda a, b: a + b, 
            [Ai * np.sin(2 * np.pi * ki * xx + 2 * np.pi * phi) +\
             Ai * np.sin(2 * np.pi * ki * yy + 2 * np.pi * phi)\
             for ki, Ai, phi in zip(freqs, amps, phase_shifts)])
    return xx, yy, zz

def make_phased_waves(amps: list[int], freqs: list[int], phase_shifts: list[int], n: int):
    """
    Creates overlapping sine waves with specified amplitudes, phase shifts, and frequencies. The domain is [0.,1.] and N points are generated.

    `amps`: List of amplitudes for the sine waves  
    `freqs`: List of frequencies for the sine waves  
    `phase_shifts`: List of phase shifts for the sine waves  
    `n`: Number of points generated for the function on [0.,1.]  

    Returns: (`t`, `yt`)  
    `t`: x-coordinates   
    `yt`: y-coordinates  
    """
    t = np.arange(0, 1, 1./n)
    yt = reduce(lambda a, b: a + b, 
                [Ai * np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, Ai, phi in zip(freqs, amps, phase_shifts)])
    return t, yt

def fft2d(zz: np.ndarray, pos_freq: bool = True, normalize: bool = True, centering: bool = False) -> np.ndarray:
    """
    Calculates 2D FFT using scipy on real input.

    `zz`: Real-valued input to compute the FFT for (2D)  
    `pos_freq`: Output positive frequencies  
    `normalize`: Normalize output  
    `centering`: Input signal is centered   

    Returns: (`zzt`)  
    `zzt`: FFT values
    """

    N = zz.shape[0]

    # Calculate FFT
    if not centering:
        zzf = scipy.fft.fft2(zz) # The frequency Y-values from FFT
    else:
        zzf = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.ifftshift(zz))) # The frequency Y-values from FFT
    #zzf = np.abs(zzf)
    zzf_real = zzf.real
    zzf_complex = zzf.imag
    if pos_freq:
        zzf_real = 2.*zzf_real[:N//2+N%2,:N//2+N%2]
        zzf_complex = 2.*zzf_complex[:N//2+N%2,:N//2+N%2]
    if normalize:
        zzf_real = zzf_real / zzf_real.shape[0]
        zzf_complex = zzf_complex / zzf_complex.shape[0]
    #xf = scipy.fft.fftfreq(N, T) # The frequency X-values from FFT
    #xf = xf[:N//2+N%2] # Extract only positive frequeny bins (x-values)
    #xxf, yyf = np.meshgrid(xf, xf)
    #zzf = np.abs(zzf[:N//2+N%2,:N//2+N%2]) # Extract only positive frequency y-values
    #zzf = 2.0/(N**2) * zzf # Multiply by 2 to account for negative frequencies (symmetric for real input), divide by number of points to normalize
    return zzf_real, zzf_complex

def fft1d(x: np.ndarray, pos_freq: bool = True, normalize: bool = True, centering: bool = False) -> np.ndarray:
    """
    Calculates 1D FFT using scipy on real input. Outputs positive frequencies (optional, on by default) normalized (optional, on by default). Input signal can be specified as centered or not (false by default).

    `x`: Real-valued input to compute the FFT  
    `pos_freq`: Output positive frequencies  
    `normalize`: Normalize output  
    `centering`: Input signal is centered   

    Returns: (`fft`, `freqs`)
    `fft`: FFT values, only positive frequencies (optional, on by default), and normalized (optional, on by default)  
    `freqs`: Frequencies of FFT  
    """

    N = x.shape[0]
    freq = scipy.fft.fftfreq(N, 1./N) # The frequency X-values from FFT
    if not centering:
        fft = np.abs(scipy.fft.fft(x))
    else:
        fft = np.abs(scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(x))))
    if pos_freq:
        freq = freq[:N//2+N%2]
        fft = 2.*fft[:N//2 + N%2]
    else:
        freq = scipy.fft.fftshift(freq)
    if normalize:
        fft = (1.0/N)*fft

    return fft, freq

def fft(yt: np.ndarray, n: int) -> tuple[Any, Any]:
    """
    Computes the Fast-Fourier Transform of an input signal.

    `yt`: Input signal  
    `n`: Number of points in the domain of our signal (? TODO: Should be on [0., 1.] anyways...)  

    Returns: (`frq`, `fftyt`)  
    `frq`: Frequencies  
    `fftyt`: FFT values at those frequencies  

    TODO: What is `T` doing?
    """
    signal_len = len(yt) # length of the signal
    k = np.arange(signal_len)
    T = signal_len/n
    frq = k/T # two sides frequency range
    frq = frq[range(signal_len//2)] # one side frequency range
    # -------------
    FFTYT = np.fft.fft(yt)/signal_len # fft computing and normalization
    FFTYT = FFTYT[range(signal_len//2)]
    fftyt = abs(FFTYT)
    return frq, fftyt

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

def power_iteration(A, num_simulations=10):
    """
    Compute the 2-norm of a matrix using the power iteration method. Theory says this should return the absolute value of the largest eigenvalue.

    `A`: Matrix to find the largest singular value squared of (TODO)
    `num_simulations`: Number of iterations of the power method  

    Returns: largest singular value of matrix A squared  

    Original notes:  
    Ideally choose a random vector
    To decrease the chance that our vector
    Is orthogonal to the eigenvector
    """
    A = A.data
    b_k = A.new(A.shape[1], 1).normal_()
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k
        # calculate the norm
        b_k1_norm = torch.norm(b_k1)
        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    return ((b_k.t() @ A @ b_k) / b_k.t() @ b_k).squeeze().abs()
    # return torch.dot(torch.dot(b_k.t(), A), b_k) / torch.dot(b_k.t(), b_k)

def spectral_norm(model): 
    """
    Compute the spectral norms of the model weights of a simple MLP.  
    
    `model`: Simple MLP to compute the spectral norms of their layers

    Returns: List of spectral norms of the layers
    """
    norms = []
    for layer in model.layers: 
        if isinstance(layer, nn.Linear):
            if layer.in_features == layer.out_features: 
                # Matrix 2-norm
                norms.append(power_iteration(layer.weight).cpu().numpy())
            elif layer.in_features == 1 or layer.out_features == 1: 
                # Vector 2-norm
                norms.append(torch.norm(layer.weight.data).cpu().numpy())
    return norms

def train_AE_model(model, lr, iters, input_, device='cpu'):
    """
    Trains a simple MLP AE

    `model`: model to train  
    `lr`: learning rate  
    `iters`: total number of iterations to run  
    `input_`: input data  
    `save_freq`: how often to save spectral norms  
    `device`: device to train on  

    Returns: TODO
    """
    # Build loss
    loss_fn = nn.MSELoss()
    # Build optim
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # Record
    frames = []
    # Create dataloader
    dl = DataLoader(input_, batch_size=1024, shuffle=True)
    # Loop! 
    model.train()
    for iter_num in tqdm(range(iters)):
        for i, data in enumerate(dl):
            data = data.to(device)
            x = data
            yt = data.view(-1, model.out_dim)
            optim.zero_grad()
            y = model(x)
            loss = loss_fn(y, yt)
            loss.backward()
            optim.step()
    # Done

    model.eval()
    return model

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
