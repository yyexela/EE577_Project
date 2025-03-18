import os
import sys
import torch
import numpy as np
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import torchvision.datasets as Datasets
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai import transforms
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import random
from skimage.transform import rotate
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError, R2Score 
from skimage import measure 
from torch.utils.data import DataLoader
import torch.nn.init as init

import logging
import csv

# python Transfer_learning_surv_trainer_v2.py

logging.basicConfig(level=logging.INFO)
# export PYTHONPATH=/home/ee577/project/src:$PYTHONPATH

src_path = os.path.abspath('src') 
# print("Absolute path to 'src':", src_path)
sys.path.append(src_path)

pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)

data_path=pkg_path+'/results/'
# print("Path to results with csvs ", data_path)
# from src import *

# Load config file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path='/home/ee577/project/Checkpoints'
# config = global_config.config
# device = torch.device(config.device) 
logging.info(f"Running on device: {device}")
direct_pairs='/home/ee577/project/training_data/DTI_AD_NC_merged_data.pkl'
# feature_model_path='/home/ee577/project/results/DTI_feat_best_model.pth'
feature_model_path='/home/ee577/project/results/DTI_feat_best_model2.pth'

with open(direct_pairs, 'rb') as f:
    X, _, Z = pickle.load(f) # images, features, survival

print(X.shape,  Z.shape)

def normalize_survival(images, survival):
    Z_log=np.log(survival)
    Z_log_norm = (Z_log - np.mean(Z_log)) / np.std(Z_log)
    t_half=np.mean(Z_log)
    lambda_value=np.log(2)/t_half
    Z_norm=np.exp(-Z_log_norm*lambda_value)
    Z_norm=(Z_norm-np.mean(Z_norm))
    valid_indices = abs(Z_norm) <= 0.3
    Z_norm = Z_norm[valid_indices]
    images = images[valid_indices]
    scaler = MinMaxScaler()
    Z_scaled = scaler.fit_transform(Z_norm.reshape(-1, 1))
    return images, Z_scaled

X, y =normalize_survival(X,Z)

def create_segmentation_mask(image, threshold=0.1):
    """
    Create a binary segmentation mask for the image based on a threshold.
    
    Args:
        image (numpy.array or torch.Tensor): The input image to segment.
        threshold (float): The intensity threshold to classify pixels as part of the region of interest.
    
    Returns:
        mask (numpy.array or torch.Tensor): The binary segmentation mask.
    """
    # Convert to numpy array if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    mask = image > threshold
    
    labeled_mask = measure.label(mask, connectivity=2)  
    mask = labeled_mask > 0  

    return torch.tensor(mask, dtype=torch.float32)

segmentation_masks = []

for i in range(X.shape[0]):  
    image = X[i, :, :, :]  
    mask = create_segmentation_mask(image, threshold=0.1)
    segmentation_masks.append(mask)

segmentation_masks = np.stack(segmentation_masks)
print(X.shape, y.shape, segmentation_masks.shape)

# Ensure X and segmentation_masks are combined and split together
X_train, X_temp, segmentation_masks_train, segmentation_masks_temp, y_train, y_temp = train_test_split(
    X, segmentation_masks, y, test_size=0.2, random_state=22)

# Split the temporary set into validation and test (70% validation, 30% test)
X_val, X_test, segmentation_masks_val, segmentation_masks_test, y_val, y_test = train_test_split(
    X_temp, segmentation_masks_temp, y_temp, test_size=0.7, random_state=22)

# Normalize target variables y_train, y_val, y_test
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)


def random_flip_rotate(image, mask, epoch):
    """
    Randomly flips and rotates the image and mask.
    
    Args:
    - image (numpy.ndarray or torch.Tensor): Input image, can be a numpy array or torch tensor.
    - mask (numpy.ndarray or torch.Tensor): Mask corresponding to the image.
    - epoch (int): Epoch number, used to set the random seed.
    - device (str): Device to move the image and mask to (default is 'cuda').
    
    Returns:
    - torch.Tensor: Transformed image and mask.
    """
    # Set the random seed using the epoch
    random.seed(epoch)
    torch.manual_seed(epoch)
    
    if isinstance(image, np.ndarray):
        image = torch.tensor(image).to(device)
        mask = torch.tensor(mask).to(device)
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
    
    if image.dim() == 4:
        # Flip horizontally with a 50% chance
        if random.random() < 0.5:
            image = image.flip(3).to(device)  # Flip along the width axis
            mask = mask.flip(3).to(device)
        
        # Rotate image and mask by 180 degrees with a 50% chance
        if random.random() < 0.5:
            image = torch.rot90(image, k=2, dims=(2, 3)).to(device)  # Rotate by 180 degrees
            mask = torch.rot90(mask, k=2, dims=(2, 3)).to(device)  # Rotate by 180 degrees
        
    else:
        raise ValueError(f"Expected image to have 3 or 4 dimensions, but got {image.dim()} dimensions.")
    
    # Remove the batch dimension if it was added earlier
    if image.dim() == 4:
        image = image.squeeze(0)
        mask = mask.squeeze(0)
    
    return image, mask

class CustomDataset(Dataset):
    def __init__(self, images, labels, masks=None, transform=None):
        """
        Args:
            images (numpy array or torch tensor): 4D tensor with shape (N, D, H, W), where N is the number of samples
            labels (numpy array or torch tensor): 2D tensor with shape (N, num_features), where N is the number of samples
            masks (numpy array or torch tensor, optional): 4D tensor with shape (N, D, H, W), where N is the number of samples
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.masks = masks  # Optional masks for segmentation
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Get the segmentation mask for this sample if provided
        if self.masks is not None:
            mask = self.masks[idx]
        else:
            mask = None

        # Convert to torch tensor if necessary
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).to(device)

        label = torch.tensor(label, dtype=torch.float32).to(device)

        # If the image is 3D (D, H, W), add channel dimension (1, D, H, W)
        if image.ndimension() == 3:  # (D, H, W)
            image = image.unsqueeze(0)  # Add channel dimension (Shape becomes: (1, D, H, W))

        # Apply transformation to the image if specified
        if self.transform:
            image = self.transform(image)

        # Return the image, label, and mask (if available)
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.float32).to(device)
                if mask.ndimension() == 3:  # (D, H, W)
                    mask = mask.unsqueeze(0)
                return {'image': image, 'label': label, 'mask': mask}
        else:
            return {'image': image, 'label': label}

class UNet3DRegression_survival(nn.Module):
    def __init__(self, in_channels, out_channels, feature_importances=None, l1_lambda=1e-11,freeze_unet=False, device='cuda'):
        super(UNet3DRegression_survival, self).__init__()

        self.device = device  # Store device info
        
        # Initialize UNet with 3D structure for segmentation task
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,  # out_channels for segmentation (e.g., 1 for binary segmentation)
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            act='PReLU',
            norm='INSTANCE',
        ).to(self.device)  # Move the UNet model to the specified device
        self.freeze_unet = freeze_unet
        # Initialize dropout and move it to device
        self.dropout_UNET = nn.Dropout3d(p=0.3).to(self.device)
        self.dropout = nn.Dropout2d(p=0.3).to(self.device)
        self.segmentation_head = nn.Conv3d(out_channels, 1, kernel_size=1).to(self.device)  # Move segmentation head to device
        if feature_importances is not None:
            self.feature_importances=feature_importances
        self.l1_lambda = l1_lambda
        # Fully connected layers for regression (survival prediction)
        self.fc1 = nn.Identity().to(self.device)
        # Initializing the regression layer as None
        self.regression_layer = nn.Identity().to(self.device)

         # Fully connected layers for regression (survival prediction)
        self.fc1=nn.Identity().to(self.device)
        self.first_size=32
        self.fc2 = nn.Linear(self.first_size, 1).to(self.device) 
        self.freeze_layers(freeze_unet)

    def freeze_layers(self, freeze_unet):
        """
        Freezes or unfreezes the UNet and regression layers based on the flag `freeze_unet`.
        """
        for param in self.unet.parameters():
            param.requires_grad = not freeze_unet  

        if not isinstance(self.regression_layer, nn.Identity):
            for param in self.regression_layer.parameters():
                param.requires_grad = not freeze_unet 

    def forward(self, x):
        if x.ndimension() == 4:  # Shape: (channels, depth, height, width)
            x = x.unsqueeze(0).to(self.device)  # Move input to device

        _, _, depth, height, width = x.size()
        if depth % 16 != 0 or height % 16 != 0 or width % 16 != 0:
            new_depth = (depth // 16 + 1) * 16
            new_height = (height // 16 + 1) * 16
            new_width = (width // 16 + 1) * 16
            x = F.interpolate(x, size=(new_depth, new_height, new_width), mode='trilinear', align_corners=True)

        x = self.unet(x)  # Forward pass through UNet
        x = self.dropout_UNET(x)  # Apply dropout
        segmentation_output = self.segmentation_head(x)  # Get segmentation output
        segmentation_output = segmentation_output.to(self.device)  # Ensure output is on the correct device
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # Global average pooling
        x = x.view(x.size(0), -1)  

        # Initialize regression_layer with the correct output size dynamically during the forward pass
        if isinstance(self.regression_layer, nn.Identity):
            self.regression_layer = nn.Linear(x.size(1), 55).to(self.device) 
        x = self.regression_layer(x)  # Apply regression layer
        self.dropout
        self.fc1 = nn.Linear(x.size(1), self.first_size).to(self.device)
        x = F.relu(self.fc1(x)) 
        self.dropout
        x = F.relu(self.fc2(x)) 
        
        survival_output = x
        return {'segmentation_output': segmentation_output, 'survival_output': survival_output}

    def custom_loss(self, segmentation_output, survival_output, target_segmentation, target_regression):
            target_segmentation = target_segmentation.to(self.device)
            target_regression = target_regression.to(self.device)

            target_size = target_segmentation.shape[2:]
            segmentation_output = F.interpolate(segmentation_output, size=target_size, mode='trilinear', align_corners=False)
            segmentation_loss = F.binary_cross_entropy_with_logits(segmentation_output, target_segmentation).to(self.device)
            loss_fn = nn.SmoothL1Loss(reduction='mean')
            regression_loss = loss_fn(survival_output, target_regression)
            l1_loss = self.l1_regularization()
            total_loss = (0.1) * segmentation_loss + regression_loss + l1_loss
            return total_loss

    def l1_regularization(self):
        """
        Computes L1 regularization (penalty) only on the fully connected layers.
        """
        l1_norm = 0.00005
        # Apply L1 regularization only to the fully connected layers (fc1, fc2, fc3, fc4)
        for name, param in self.named_parameters():
            if name in ['fc1.weight', 'fc2.weight']:  # Check if it's a fully connected layer
                l1_norm += torch.sum(torch.abs(param))  # L1 regularization on weights
        return self.l1_lambda * l1_norm 

def save_checkpoint(model, optimizer, epoch, loss, batch=-1, val_loss=None, path='/home/ee577/project/results/Model_surv.pth', csv_path='/home/ee577/project/results/Loss_surv.csv', attention_model=None):
    try:
        # Save checkpoint to the .pth file
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss,
            'val_loss': val_loss,
            'batch': batch,
        }

        # If attention model is provided, save its state dict as well
        if attention_model is not None:
            checkpoint['attention_model_state_dict'] = attention_model.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
        # Handle CSV file writing
        # Check if the CSV file exists
        if not os.path.isfile(csv_path):
            print(f"CSV file not found. Creating a new one: {csv_path}")
            # If it doesn't exist, create the file and write the header
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss'])  # Add header row

        # Append new data to the CSV file
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # If val_loss is None, write a placeholder "N/A" or some default value
            if val_loss is None:
                val_loss = 'N/A'  # Set val_loss to 'N/A' if None
            writer.writerow([epoch, batch, loss, val_loss])  # Append data

        print(f"Checkpoint data saved to CSV at {csv_path}")

    except Exception as e:
        print(f"Error occurred while saving checkpoint: {str(e)}")

def add_noise_to_labels(labels, noise_factor=0.05, seed=None):
    """
    Adds or subtracts noise to labels randomly, based on the seed.
    
    Args:
        labels (Tensor): The tensor of labels to which noise is to be added or subtracted.
        noise_factor (float): The factor by which the noise is scaled (default is 0.1).
        seed (int, optional): A seed for randomness to control noise addition or subtraction. If None, random behavior.
        
    Returns:
        Tensor: Labels with added or subtracted noise.
    """
    # Set the seed if provided to control the randomness
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.randn_like(labels) * noise_factor  # Generate Gaussian noise
    
    # Randomly decide to add or subtract noise
    if torch.randint(0, 2, (1,)).item() == 0:  # Randomly choose 0 or 1
        return labels + noise  # Add noise
    else:
        return labels - noise 

def train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        num_epochs=1000, 
        patience=10, 
        eval_every=2,  
        max_grad_norm = 4.0,
        last_epoch=None
        ):
    model.to(device)
    model.train()

    best_val_loss = float('inf')  # Initialize with a large number
    batches_since_improvement = 0  # Count how many epochs since last improvement
    val_losses = []  # List to store the validation losses
    path = f'/home/ee577/project/results/DTI_surv_checkpoint.pth'
    if last_epoch:
        epoch_range=range(last_epoch, num_epochs)
    else:
        epoch_range=range(num_epochs)

    for epoch in epoch_range:
        random.seed(epoch)
        np.random.seed(epoch)
        torch.manual_seed(epoch)

        # Iterate over batches in the train_loader
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels,segmentation_masks = batch['image'], batch['label'], batch['mask']

            inputs = inputs.to(device)
            labels = labels.to(device)
            segmentation_masks = segmentation_masks.to(device)

            transformed_inputs = []
            transformed_masks= []

            # Apply random transformations (flip, rotate, noise) to the entire batch
            for i in range(inputs.size(0)):
                image = inputs[i].cpu().numpy()
                mask = segmentation_masks[i].cpu().numpy()
                if random.random() < 0.5:
                    image, mask = random_flip_rotate(image, mask, epoch)        
                # Ensure image is 4D (1, D, H, W)
                image = torch.tensor(image, dtype=torch.float32).to(device)
                mask = torch.tensor(mask, dtype=torch.float32).to(device)
                if image.ndimension() == 3:  # (D, H, W) -> Add channel dimension
                    image = image.unsqueeze(0)  # Now it becomes (1, D, H, W)
                    mask = mask.unsqueeze(0)
                transformed_inputs.append(image)
                transformed_masks.append(mask)

            # Now, stack the images after ensuring all have the shape (1, D, H, W)
            transformed_inputs = torch.stack(transformed_inputs).to(device)
            transformed_masks = torch.stack(transformed_masks).to(device)

            if model.training:  # Add noise only during training
                noisy_labels = add_noise_to_labels(labels, seed=epoch)  
            optimizer.zero_grad()

            # Forward pass
            outputs = model(transformed_inputs)
            # Calculate segmentation loss using the segmentation masks
            segmentation_output = outputs['segmentation_output']
            survival_output = outputs['survival_output']
  
            # Use weighted MSE loss
            train_loss = model.custom_loss(segmentation_output,survival_output, transformed_masks, labels)  
            l1_loss = model.l1_regularization()

            total_loss = train_loss + l1_loss
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Average loss for this epoch

            train_loss = train_loss.item()
            val_loss = evaluate_validation_loss(model, val_loader) # model in eval mode
            logging.info(f"Epoch {epoch + 1}/{num_epochs},  Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")
            print(f"Epoch {epoch + 1}/{num_epochs},Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")
            
            model.train() # re-enabled gradients
            # Perform validation every `eval_every` epochs
            if (batch_idx + 1) % eval_every == 0:
                scheduler.step(val_loss)
                val_losses.append(val_loss)  # Save the validation loss
                save_checkpoint(model, optimizer, epoch=epoch, loss=train_loss,val_loss =val_loss, batch=batch_idx, path=path)
                # Early stopping: Check if validation loss has improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    batches_since_improvement = 0
                    # Save model if validation loss improves
                    torch.save(model.state_dict(), f"best_model.pth")
                else:
                    batches_since_improvement += 1
                # Early stopping condition
                if batches_since_improvement >= patience:
                    print(f"Early stopping after {epoch + 1} batches, validation loss has not improved for {patience} epochs.")
                    break
        # Clean up
        del inputs, labels, transformed_inputs
        torch.cuda.empty_cache()

def evaluate_validation_loss(model, val_loader):
    """
    Evaluate the model on the validation set and compute the validation loss.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels,segmentation_masks = batch['image'], batch['label'],batch['mask']
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            segmentation_output = outputs['segmentation_output']
            regression_output = outputs['survival_output']

            # Calculate loss
            loss = model.custom_loss(segmentation_output,regression_output, segmentation_masks,labels)
            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_val_loss = val_loss / total_samples
    return avg_val_loss


batch_siz=16
train_loader = DataLoader(CustomDataset(X_train, y_train, masks=segmentation_masks_train), batch_size=batch_siz, shuffle=False)
val_loader = DataLoader(CustomDataset(X_val, y_val, masks=segmentation_masks_val), batch_size=batch_siz, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test, y_test, masks=segmentation_masks_test), batch_size=batch_siz, shuffle=False)

# Example of how you can iterate through the dataset
for batch in train_loader:
    images = batch['image']
    labels = batch['label']
    masks = batch['mask']

n_shape = batch['image'].shape 
out_shape=y_train.shape[1:][0]

def load_model(model, feature_model_path, linear_model_path=None):
    state_dict = torch.load(feature_model_path)
    model.load_state_dict(state_dict, strict=False)
    if linear_model_path:
        linear_state_dict = torch.load(linear_model_path)
        model_state_dict = model.state_dict()
        linear_layer_keys = [key for key in linear_state_dict.keys() if 'fc' in key]  
        for key in linear_layer_keys:
            if key in model_state_dict:
                model_state_dict[key] = linear_state_dict[key]
        model.load_state_dict(model_state_dict)
    model.eval()
    
    return model

model = UNet3DRegression_survival(in_channels=1, out_channels=out_shape, device=device) # feature_importances=init_weights
DTI_model = load_model(model, feature_model_path).to(device)

optimizer = Adam([
    {'params': model.unet.parameters(), 'weight_decay': 1e-10},  # Less regularization on U-net layers
    {'params': model.regression_layer.parameters(), 'weight_decay': 1e-9},  # Regularization for regression layers
    {'params': model.fc1.parameters(), 'weight_decay': 0.001},
    {'params': model.fc2.parameters(), 'weight_decay': 0.001},
], lr=1e-6)

from torch.optim.lr_scheduler import LambdaLR

# Warm-up strategy for first 5 epochs
def lr_lambda(epoch):
    return min(1.0, (epoch + 1) / 5.0)  # Gradually increase from 0 to 1

warmup_scheduler = LambdaLR(optimizer, lr_lambda)


warmup_scheduler.step()
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

train_model(DTI_model, train_loader, val_loader, optimizer,warmup_scheduler, num_epochs=1000, patience=10, eval_every=10).to(device)