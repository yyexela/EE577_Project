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
from torch.optim import Adam
import random
from skimage.transform import rotate
from skimage.util import random_noise
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError, R2Score 
from skimage import measure 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import logging

logging.basicConfig(level=logging.INFO)
# export PYTHONPATH=/home/ee577/project/src:$PYTHONPATH

src_path = os.path.abspath('src') 
print("Absolute path to 'src':", src_path)
sys.path.append(src_path)

pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)

data_path=pkg_path+'/project/results/'
print("Path to results with csvs ", data_path)
# from src import *

# Load config file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = global_config.config
# device = torch.device(config.device) 
logging.info(f"Running on device: {device}")
direct_pairs='/home/ee577/project/results/DTI_AD_NC_paired_data.pkl'

with open(direct_pairs, 'rb') as f:
    X, y = pickle.load(f)

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
    X, segmentation_masks, y, test_size=0.2, random_state=1)

# Split the temporary set into validation and test (70% validation, 30% test)
X_val, X_test, segmentation_masks_val, segmentation_masks_test, y_val, y_test = train_test_split(
    X_temp, segmentation_masks_temp, y_temp, test_size=0.7, random_state=1)

# Normalize target variables y_train, y_val, y_test
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)

def random_flip_and_rotate(image, mask, epoch):
    """
    Randomly flips and rotates the image and the mask, ensuring both are flipped and rotated the same way.
    The random seed is based on the given epoch number to introduce variability across epochs.
    
    Args:
    - image (numpy.ndarray or torch.Tensor): Input image, can be a numpy array or torch tensor.
    - mask (numpy.ndarray or torch.Tensor): Input mask, can be a numpy array or torch tensor.
    - epoch (int): The epoch number, used to set the random seed.
    
    Returns:
    - torch.Tensor: Randomly flipped and rotated image and mask (both flipped and rotated the same way).
    """
    # Set the random seed based on epoch for reproducibility
    random.seed(epoch)
    torch.manual_seed(epoch)

    # Save the original device for later use
    device = image.device if isinstance(image, torch.Tensor) else 'cpu'

    # If the image is a numpy array, convert to tensor
    if isinstance(image, np.ndarray):
        image = torch.tensor(image, dtype=torch.float32).to(device)  # Move to the same device
        mask = torch.tensor(mask, dtype=torch.float32).to(device)  # Move to the same device
    
    # Add a batch dimension if the image has 3 dimensions (D, H, W) -> (1, D, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
    
    # (batch_size, channels, height, width) -> Flip and rotate
    if image.dim() == 4:
        # Random horizontal and vertical flip
        flip_horizontally = random.random() < 0.5
        flip_vertically = random.random() < 0.5
        
        # Apply the same flip to both image and mask
        if flip_horizontally:
            image = image.flip(3)  # Flip along the width axis
            mask = mask.flip(3)    # Flip along the width axis
        
        if flip_vertically:
            image = image.flip(2)  # Flip along the height axis
            mask = mask.flip(2)    # Flip along the height axis
        
        # Random rotation angle (in degrees)
        rotation_angle = random.choice([0, 90, 180, 270])  # Choose from 0, 90, 180, or 270 degrees
        
        # Apply the same rotation to both the image and the mask
        image = TF.rotate(image, rotation_angle)
        mask = TF.rotate(mask, rotation_angle)
    
    else:
        raise ValueError(f"Expected image to have 3 or 4 dimensions, but got {image.dim()} dimensions.")
    
    # If we added a batch dimension earlier (for single image), remove it now
    if image.dim() == 4:
        image = image.squeeze(0)
        mask = mask.squeeze(0)

    return image.to(device), mask.to(device)


class UNet3DRegression_survival(nn.Module):
    def __init__(self, in_channels, out_channels, feature_importances=None, l1_lambda=1e-5):
        super(UNet3DRegression_survival, self).__init__()

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
        )

        # Initialize regression layer as None initially
        self.regression_layer = None
        self.feature_importances = feature_importances  # Holds feature importance for scaling weights
        self.l1_lambda = l1_lambda  # L1 regularization strength

        # Add layers for survival prediction (5 bins for classification)
        self.fc1 = nn.Linear(55, 128).to(device)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64).to(device)   # Intermediate layer
        self.fc3 = nn.Linear(64, 32).to(device)    # Intermediate layer
        self.fc4 = nn.Linear(32, 5).to(device)     # Final layer for 5 survival bins

    def forward(self, x):
        # Check for 5D input shape (batch_size, channels, depth, height, width)
        if x.ndimension() == 4:  # Shape: (channels, depth, height, width)
            x = x.unsqueeze(0)

        _, _, depth, height, width = x.size()

        # Resize input dimensions if necessary
        if depth % 16 != 0 or height % 16 != 0 or width % 16 != 0:
            new_depth = (depth // 16 + 1) * 16
            new_height = (height // 16 + 1) * 16
            new_width = (width // 16 + 1) * 16
            x = F.interpolate(x, size=(new_depth, new_height, new_width), mode='trilinear', align_corners=True)

        # Apply the U-Net
        x = self.unet(x)

        # Flatten the output for the regression layer
        flattened_size = x.numel() // x.size(0)
        if self.regression_layer is None:
            self.regression_layer = nn.Linear(flattened_size, 55).to(device)   # Adjust output size to 55 parameters

        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.regression_layer(x)  # Regression output (55 parameters)

        # Apply feature importance weighting if available
        if self.feature_importances is not None:
            x = x * self.feature_importances  # Element-wise multiplication with feature importance

        # Pass through fully connected layers for survival prediction
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = F.relu(self.fc2(x))  # Second fully connected layer
        x = F.relu(self.fc3(x))  # Third fully connected layer
        x = self.fc4(x)  # Final output layer (5 survival bins)

        return x

    def l1_regularization(self):
        """
        Computes L1 regularization (penalty) on the model parameters.
        """
        l1_norm = 0.0
        for param in self.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_norm  # L1 penalty term scaled by lambda


def save_checkpoint(model, optimizer, epoch, loss,batch=-1, val_loss=None, path='/home/ee577/project/results/checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss,
        'val_loss': val_loss,
        'batch':batch,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        num_epochs=1000, 
        patience=10, 
        eval_every=2,  
        seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        max_grad_norm = 3.0,
        segmentation_weight=0.5,  # You can tune this
        classification_weight=1.0  # You can tune this
        ):

    model.to(device)
    model.train()
    best_val_loss = float('inf')  # Initialize with a large number
    batches_since_improvement = 0  # Count how many epochs since last improvement
    val_losses = []  # List to store the validation losses

    # CrossEntropyLoss for classification
    criterion_classification = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        seed = epoch  # Cycle through the seeds if num_epochs > len(seeds)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        running_loss = 0.0

        # Iterate over batches in the train_loader
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels, segmentation_masks = batch['image'], batch['label'], batch['mask']

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
                    image, mask = random_flip(image, mask, seed)
    
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
            segmentation_output = outputs['segmentation_output']  # Segmentation output from U-Net
            regression_output = outputs['regression_output']  # Classification output

            # Compute segmentation loss (e.g., Dice Loss or CrossEntropyLoss)
            segmentation_loss = criterion_classification(segmentation_output, segmentation_masks)

            # Compute classification loss (e.g., CrossEntropyLoss for 5 bins)
            classification_loss = criterion_classification(regression_output, labels)

            # Combine both losses
            total_loss = segmentation_weight * segmentation_loss + classification_weight * classification_loss

            # Apply L1 regularization
            l1_loss = model.l1_regularization()

            total_loss += l1_loss
            total_loss.backward()

            clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            # Average loss for this epoch
            train_loss = total_loss.item()
            val_loss = evaluate_validation_loss(model, val_loader)  # model in eval mode
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Seed {seed}, Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Seed {seed}, Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")
            
            model.train()  # Re-enable gradients
            
            # Perform validation every `eval_every` epochs
            if (batch_idx + 1) % eval_every == 0:
                scheduler.step(val_loss)
                val_losses.append(val_loss)  # Save the validation loss
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
        #path = f'/home/ee577/project/results/DTI_feat_{epoch}.pth'
        save_checkpoint(model, optimizer, epoch='latest', loss=train_loss, val_loss=val_loss, path=path)

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
            inputs, labels, segmentation_masks = batch['image'], batch['label'], batch['mask']
            inputs = inputs.to(device)
            labels = labels.to(device)
            segmentation_masks = segmentation_masks.to(device)

            # Forward pass
            logits = model(inputs)  # Raw logits from the model

            # Calculate classification loss (CrossEntropyLoss expects logits)
            loss = criterion(logits, labels)
            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_val_loss = val_loss / total_samples
    return avg_val_loss


batch_siz=16
train_loader = DataLoader(CustomDataset(X_train, y_train, masks=segmentation_masks_train), batch_size=batch_siz, shuffle=True)
val_loader = DataLoader(CustomDataset(X_val, y_val, masks=segmentation_masks_val), batch_size=batch_siz, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test, y_test, masks=segmentation_masks_test), batch_size=batch_siz, shuffle=False)

# Example of how you can iterate through the dataset
for batch in train_loader:
    images = batch['image']
    labels = batch['label']
    masks = batch['mask']

in_shape = batch['image'].shape 
out_shape=y_train.shape[1:][0]

model = UNet3DRegression(in_channels=1, out_channels=out_shape, weights=shapley_values_scaled_tensor)

optimizer = Adam([
    {'params': model.unet.parameters(), 'weight_decay': 1e-10},  # Less regularization on U-net layers
    {'params': model.regression_layer.parameters(), 'weight_decay': 1e-9},  # L2 regularization for regression layer
], lr=1e-7)
# Learning Rate Scheduler (Reduce learning rate when validation loss plateaus)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

def load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
model = UNet3DRegression_survival(in_channels=1, out_channels=out_shape)
DTI_feat_model = load_model(model, data_path+f"DTI_feat_{0}.pth")

# DTI_feat_model = load_model(model, data_path+f"DTI_feat_{0}.pth")
seeds = [21, 22, 23, 24, 25, 26, 27, 28, 29, 20]  # The list of seeds to iterate over
train_model(model, train_loader, val_loader, optimizer,scheduler, num_epochs=1000, patience=10, eval_every=10).to(device)