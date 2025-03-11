import os
import sys
import torch
import numpy as np
import torch.nn as nn
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
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import random
import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError, R2Score 
from skimage import measure 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)

data_path=pkg_path+'/results/'

from src import *

# Load config file
config = global_config.config
device = torch.device(config.device) 
direct_pairs='/home/ee577/project/results/DTI_AD_NC_paired_data.pkl'

with open(direct_pairs, 'rb') as f:
    X, y = pickle.load(f)

import numpy as np
import torch
from skimage import measure

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


def random_flip(image):
    """
    Randomly flips the image horizontally or vertically.
    
    Args:
    - image (numpy.ndarray or torch.Tensor): Input image, can be a numpy array or torch tensor.
    
    Returns:
    - torch.Tensor: Randomly flipped image.
    """
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # (batch_size, channels, height, width)
    if image.dim() == 4:
        # Flip horizontally (along the width axis)
        if random.random() < 0.5:
            image = image.flip(3)  
        
        # Flip vertically (along the height axis)
        if random.random() < 0.5:
            image = image.flip(2)  
    else:
        raise ValueError(f"Expected image to have 3 or 4 dimensions, but got {image.dim()} dimensions.")
    
    # If we added a batch dimension earlier (for single image), remove it now
    if image.dim() == 4:
        image = image.squeeze(0)
    
    return image

def add_random_noise(image, noise_factor=0.05):
    """
    Adds random noise to the image.
    """
    noise = np.random.normal(0, noise_factor, image.shape)
    return image + noise
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
            image = torch.tensor(image, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.float32)

        # If the image is 3D (D, H, W), add channel dimension (1, D, H, W)
        if image.ndimension() == 3:  # (D, H, W)
            image = image.unsqueeze(0)  # Add channel dimension (Shape becomes: (1, D, H, W))

        # Apply transformation to the image if specified
        if self.transform:
            image = self.transform(image)

        # Return the image, label, and mask (if available)
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.float32)
                if mask.ndimension() == 3:  # (D, H, W)
                    mask = mask.unsqueeze(0)
                return {'image': image, 'label': label, 'mask': mask}
        else:
            return {'image': image, 'label': label}
class UNet3DRegression(nn.Module):
    def __init__(self, in_channels, out_channels, weights,segmentation_weight=0.8, l1_lambda=1e-10):
        super(UNet3DRegression, self).__init__()

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
        self.regression_layer = nn.Identity()
        sum_weights=sum(weights)
        self.weights = weights/sum_weights
        self.dropout = nn.Dropout3d(p=0.3)
        self.l1_lambda = l1_lambda
        self.segmentation_weight=segmentation_weight
        self.segmentation_head = nn.Conv3d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        if x.ndimension() == 4:  # Shape: (channels, depth, height, width)
            x = x.unsqueeze(0)
        
        _, _, depth, height, width = x.size()
        if depth % 16 != 0 or height % 16 != 0 or width % 16 != 0:
            new_depth = (depth // 16 + 1) * 16
            new_height = (height // 16 + 1) * 16
            new_width = (width // 16 + 1) * 16
            x = F.interpolate(x, size=(new_depth, new_height, new_width), mode='trilinear', align_corners=True)

        x = self.unet(x)
        x = self.dropout(x)
        segmentation_output = self.segmentation_head(x)

        # Apply global average pooling to reduce the spatial dimensions
        # The output shape will be (batch_size, channels, 1, 1, 1)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # Global average pooling
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, channels]

        # Initialize regression_layer with the correct output size dynamically during the forward pass
        if isinstance(self.regression_layer, nn.Identity):
            self.regression_layer = nn.Linear(x.size(1), 55)  # Adjust output to 55

        regression_output = self.regression_layer(x)  # Apply regression layer (55 output features)

        # print(f"Regression tensor size: {regression_output.size()}")
        return {'segmentation_output': segmentation_output, 'regression_output': regression_output}
    
    def weighted_mse_loss(self, segmentation_output, regression_output, target_segmentation, target_regression):
        """
        Combines the segmentation loss and regression loss into a single loss function.
        Args:
            segmentation_output (Tensor): The output of the model for segmentation.
            regression_output (Tensor): The output of the model for regression.
            target_segmentation (Tensor): The ground truth segmentation.
            target_regression (Tensor): The ground truth regression labels.
        """
        
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
        max_grad_norm = 5.0
        ):
    model.to(device)
    model.train()

    best_val_loss = float('inf')  # Initialize with a large number
    batches_since_improvement = 0  # Count how many epochs since last improvement
    val_losses = []  # List to store the validation losses

    for epoch in range(num_epochs):
        seed = seeds[epoch % len(seeds)]  # Cycle through the seeds if num_epochs > len(seeds)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        running_loss = 0.0

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
                    image = random_flip(image)
                    mask=random_flip(mask)
                if random.random() < 0.5:
                    image = add_random_noise(image)
                    
                # Ensure image is 4D (1, D, H, W)
                image = torch.tensor(image, dtype=torch.float32)
                mask = torch.tensor(mask, dtype=torch.float32)
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
            print(f"Epoch {epoch + 1}/{num_epochs}, Seed {seed}, Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")
            path = f'/home/ee577/project/results/DTI_feat_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch='latest', loss=train_loss,val_loss =val_loss, batch=batch_idx, path=path)
            model.train() # re-enabled gradients
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

train_loader = DataLoader(CustomDataset(X_train, y_train, masks=segmentation_masks_train), batch_size=32, shuffle=True)
val_loader = DataLoader(CustomDataset(X_val, y_val, masks=segmentation_masks_val), batch_size=32, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test, y_test, masks=segmentation_masks_test), batch_size=32, shuffle=False)

# Example of how you can iterate through the dataset
for batch in train_loader:
    images = batch['image']
    labels = batch['label']
    masks = batch['mask']
    
    # Do something with images, labels, and masks...
    print(images.shape, labels.shape, masks.shape if masks is not None else "No Mask")

# used to weight the loss values
shapley_df = pd.read_csv(data_path + 'shap_DTI_AD_NC_.csv')
shapley_df = shapley_df[shapley_df['Summed_Shapy_Values'] >= 200]
print(f"Original shape: {shapley_df.shape}") 
new_values = [200, 200, 200, 400, 400]
new_values_df = pd.DataFrame(new_values, columns=['Summed_Shapy_Values'])
shapley_df = pd.concat([shapley_df, new_values_df], ignore_index=True)
shapley_df=shapley_df['Summed_Shapy_Values'].values
print(f"Updated shape: {shapley_df.shape}")
shapley_values = np.array(shapley_df).reshape(-1, 1) 
print(f"Updated shape: {shapley_values.shape}")
scaler = MinMaxScaler()
shapley_values_scaled = scaler.fit_transform(shapley_values)
shapley_values_scaled_tensor = torch.tensor(shapley_values_scaled, dtype=torch.float32)

in_shape = batch['image'].shape 
out_shape=y_train.shape[1:][0]

model = UNet3DRegression(in_channels=1, out_channels=out_shape, weights=shapley_values_scaled_tensor)

optimizer = Adam([
    {'params': model.unet.parameters(), 'weight_decay': 1e-10},  # Less regularization on U-net layers
    {'params': model.regression_layer.parameters(), 'weight_decay': 1e-9},  # L2 regularization for regression layer
], lr=1e-7)
# Learning Rate Scheduler (Reduce learning rate when validation loss plateaus)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

# DTI_feat_model = load_model(model, data_path+f"DTI_feat_{0}.pth")
seeds = [21, 22, 23, 24, 25, 26, 27, 28, 29, 20]  # The list of seeds to iterate over
train_model(model, train_loader, val_loader, optimizer,scheduler, num_epochs=1000, patience=10, eval_every=10).to(device)
