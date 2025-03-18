import os
import sys
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torchvision.datasets as Datasets
import torchvision.transforms as T
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
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import random
import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError, R2Score 
from skimage import measure 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from skimage import measure
import csv
import logging
logging.basicConfig(level=logging.INFO)


pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)

data_path=pkg_path+'/results/'

checkpoint_path='/home/ee577/project/Checkpoints' 

from src import *

# Load config file
# config = global_config.config
# device = torch.device(config.device) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on this device {device}")
logging.info(f"Running on device: {device}")

direct_pairs='/home/ee577/project/training_data/DSC_ED_merged_data.pkl'

with open(direct_pairs, 'rb') as f:
    X, y, _ = pickle.load(f) # image, features, survival

nan_count_per_column = np.sum(np.isnan(y), axis=0)  
columns_to_keep = nan_count_per_column <= (0.3 * y.shape[0])  
y = y[:, columns_to_keep] 
y = np.nan_to_num(y, nan=0)
print(y.shape)


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

# Ensure X and segmentation_masks are combined and split together
X_train, X_temp, segmentation_masks_train, segmentation_masks_temp, y_train, y_temp = train_test_split(
    X, segmentation_masks, y, test_size=0.2, random_state=42)

# Split the temporary set into validation and test (70% validation, 30% test)
X_val, X_test, segmentation_masks_val, segmentation_masks_test, y_val, y_test = train_test_split(
    X_temp, segmentation_masks_temp, y_temp, test_size=0.7, random_state=42)

# Normalize target variables y_train, y_val, y_test
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)

print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}, segmentation_masks_train={segmentation_masks_train.shape}")
print(f"Validation set: X_val={X_val.shape}, y_val={y_val.shape}, segmentation_masks_val={segmentation_masks_val.shape}")
print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}, segmentation_masks_test={segmentation_masks_test.shape}")


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



class UNet3DRegression(nn.Module):
    def __init__(self, in_channels, out_channels, weights, segmentation_weight=0.8, l1_lambda=1e-11, device=device):
        super(UNet3DRegression, self).__init__()

        self.device = device  # Set the device for the model
        # Initialize UNet with 3D structure
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            act='PReLU',
            norm='INSTANCE',
            dropout=0.1,
        ).to(device)  # Move UNet to the device

        # Initialize regression layer as None initially
        self.regression_layer = nn.Identity().to(device)
        
        # Normalize and move weights to the correct device
        sum_weights = sum(weights)
        self.weights = (weights / sum_weights).to(device)

        # Initialize dropout and move it to device
        self.dropout = nn.Dropout3d(p=0.3).to(device)

        # Initialize segmentation weight and move it to device
        self.segmentation_weight = torch.tensor(segmentation_weight).to(device)  # Ensure segmentation weight is a tensor
        self.segmentation_head = nn.Conv3d(out_channels, 1, kernel_size=1).to(device)  # Move segmentation head to device

        self.l1_lambda = l1_lambda  # L1 regularization lambda

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
        x = self.dropout(x)  # Apply dropout
        segmentation_output = self.segmentation_head(x)  # Get segmentation output
        segmentation_output = segmentation_output.to(self.device)  # Ensure output is on the correct device

        # Apply global average pooling to reduce the spatial dimensions
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # Global average pooling
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, channels]

        # Initialize regression_layer with the correct output size dynamically during the forward pass
        if isinstance(self.regression_layer, nn.Identity):
            self.regression_layer = nn.Linear(x.size(1), 55).to(self.device)  # Adjust output to 55 and move to device

        regression_output = self.regression_layer(x)  # Apply regression layer (55 output features)
        regression_output = regression_output.to(self.device)  # Ensure output is on the correct device

        return {'segmentation_output': segmentation_output, 'regression_output': regression_output}

    def weighted_mse_loss(self, segmentation_output, regression_output, target_segmentation, target_regression):
        # Ensure all targets are on the same device as the model outputs
        target_segmentation = target_segmentation.to(self.device)
        target_regression = target_regression.to(self.device)

        # Resize the segmentation output to match the size of the target segmentation
        target_size = target_segmentation.shape[2:]  # [74, 98, 86]
        segmentation_output = F.interpolate(segmentation_output, size=target_size, mode='trilinear', align_corners=False)

        # Calculate the binary cross-entropy loss for segmentation
        segmentation_loss = F.binary_cross_entropy_with_logits(segmentation_output, target_segmentation)

        # Ensure the regression output and target regression have the same shape
        assert regression_output.size(1) == target_regression.size(1), \
            f"Regression output size {regression_output.size(1)} does not match target size {target_regression.size(1)}"
        
        # Calculate the MSE loss for regression
        regression_loss = (regression_output - target_regression) ** 2
        weighted_mse_loss = regression_loss * self.weights.T 
        regression_loss = weighted_mse_loss.mean()

        # Combine both losses
        total_loss = segmentation_loss + regression_loss
        return total_loss

    def l1_regularization(self):
        """
        Computes L1 regularization (penalty) on the model parameters.
        """
        l1_norm = 0.0
        for param in self.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_norm  # L1 penalty term scaled by lambda

def save_checkpoint(model, optimizer, epoch, loss, batch=-1, val_loss=None, path='/home/ee577/project/Checkpoint/checkpoint.pth', csv_path='/home/ee577/project/results/checkpoints.csv'):
    # Save checkpoint to the .pth file
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss,
        'val_loss': val_loss,
        'batch': batch,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

    # Check if the CSV file exists
    if not os.path.isfile(csv_path):
        # If it doesn't exist, create the file and write the header
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss'])  # Add header row
        print(f"CSV file created at {csv_path}")

    # Append new data to the CSV file
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, batch, loss, val_loss])  # Append data
    print(f"Checkpoint data saved to CSV at {csv_path}")

def train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        num_epochs=1000, 
        patience=10, 
        eval_every=2,  
        seeds=0,
        max_grad_norm = 5.0,
        last_epoch=None
        ):
    model.to(device)
    model.train()

    best_val_loss = float('inf')  # Initialize with a large number
    batches_since_improvement = 0  # Count how many epochs since last improvement
    val_losses = []  # List to store the validation losses
    path = f'/home/ee577/project/results/DSC_feat_checkpoint.pth'
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

            optimizer.zero_grad()

            # Forward pass
            outputs = model(transformed_inputs)
            # Calculate segmentation loss using the segmentation masks
            segmentation_output = outputs['segmentation_output']
            regression_output = outputs['regression_output']

            # Use weighted MSE loss
            train_loss = model.weighted_mse_loss(segmentation_output,regression_output, transformed_masks,labels)  
            l1_loss = model.l1_regularization()

            total_loss = train_loss + l1_loss
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Average loss for this epoch

            train_loss = train_loss.item()
            val_loss = evaluate_validation_loss(model, val_loader) # model in eval mode
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")
            
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
            regression_output = outputs['regression_output']

            # Calculate loss
            loss = model.weighted_mse_loss(segmentation_output,regression_output, segmentation_masks,labels)
            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_val_loss = val_loss / total_samples
    return avg_val_loss

batch_size=16

train_loader = DataLoader(CustomDataset(X_train, y_train, masks=segmentation_masks_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomDataset(X_val, y_val, masks=segmentation_masks_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test, y_test, masks=segmentation_masks_test), batch_size=batch_size, shuffle=False)

# Example of how you can iterate through the dataset
for batch in train_loader:
    images = batch['image']
    labels = batch['label']
    masks = batch['mask']
    


# used to weight the loss values
shapley_df = pd.read_csv(data_path + 'shap_DSC_ap_rCBV_ED.csv')
shapley_df = shapley_df[shapley_df['Summed_Shapy_Values'] >= 200]
new_values = [400, 400] #200, 200,
new_values_df = pd.DataFrame(new_values, columns=['Summed_Shapy_Values'])
shapley_df = pd.concat([shapley_df, new_values_df], ignore_index=True)
shapley_df=shapley_df['Summed_Shapy_Values'].values
shapley_values = np.array(shapley_df).reshape(-1,1) 
scaler = MinMaxScaler()
shapley_values_scaled = scaler.fit_transform(shapley_values)
shapley_values_scaled_tensor = torch.tensor(shapley_values_scaled, dtype=torch.float32)

in_shape = batch['image'].shape 
out_shape=y_train.shape[1:][0]

model = UNet3DRegression(in_channels=1, out_channels=out_shape, weights=shapley_values_scaled_tensor)

optimizer = Adam([
    {'params': model.unet.parameters(), 'weight_decay': 1e-11},  # Less regularization on U-net layers
    {'params': model.regression_layer.parameters(), 'weight_decay': 1e-9},  # L2 regularization for regression layer
], lr=1e-7)
# Learning Rate Scheduler (Reduce learning rate when validation loss plateaus)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

# DTI_feat_model = load_model(model, data_path+f"DTI_feat_{0}.pth")

train_model(model, train_loader, val_loader, optimizer,scheduler, num_epochs=1000, patience=10, eval_every=10).to(device)
