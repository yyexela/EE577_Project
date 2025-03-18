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
DSC_mod_path='/home/ee577/project/best_models_unet/DSC_surv_best_model.pth'
DTI_mod_path='/home/ee577/project/best_models_unet/DTI_surv_best_model.pth'

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



# Ensure X and segmentation_masks are combined and split together
X_train, X_temp, y_train, y_temp = train_test_split(
    X,  y, test_size=0.2, random_state=22)

# Split the temporary set into validation and test (70% validation, 30% test)
X_val, X_test,  y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.7, random_state=22)

# Normalize target variables y_train, y_val, y_test
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)

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
        ).to(self.device) 
        self.fc1 = nn.Identity().to(self.device)
        self.regression_layer = nn.Identity().to(self.device)
        self.fc1=nn.Identity().to(self.device)
        self.first_size=32

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
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # Global average pooling
        x = x.view(x.size(0), -1)  

        # Initialize regression_layer with the correct output size dynamically during the forward pass
        if isinstance(self.regression_layer, nn.Identity):
            self.regression_layer = nn.Linear(x.size(1), 55).to(self.device) 
        x = self.regression_layer(x)  # Apply regression layer
        self.fc1 = nn.Linear(x.size(1), self.first_size).to(self.device)
        x = F.relu(self.fc1(x)) 
        survival_output = x
        return survival_output

def random_flip_rotate(image,  epoch):
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

    
    if image.dim() == 3:
        image = image.unsqueeze(0)

    
    if image.dim() == 4:
        # Flip horizontally with a 50% chance
        if random.random() < 0.5:
            image = image.flip(3).to(device)  # Flip along the width axis
           
        
        # Rotate image and mask by 180 degrees with a 50% chance
        if random.random() < 0.5:
            image = torch.rot90(image, k=2, dims=(2, 3)).to(device)  # Rotate by 180 degrees
           
    else:
        raise ValueError(f"Expected image to have 3 or 4 dimensions, but got {image.dim()} dimensions.")
    
    # Remove the batch dimension if it was added earlier
    if image.dim() == 4:
        image = image.squeeze(0)
      
    
    return image

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        """
        Args:
            images (numpy array or torch tensor): 4D tensor with shape (N, D, H, W), where N is the number of samples
            labels (numpy array or torch tensor): 2D tensor with shape (N, num_features), where N is the number of samples
            masks (numpy array or torch tensor, optional): 4D tensor with shape (N, D, H, W), where N is the number of samples
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]


        # Convert to torch tensor if necessary
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).to(device)

        label = torch.tensor(label, dtype=torch.float32).to(device)

        # If the image is 3D (D, H, W), add channel dimension (1, D, H, W)
        if image.ndimension() == 3:  # (D, H, W)
            image = image.unsqueeze(0)  # Add channel dimension (Shape becomes: (1, D, H, W))
        return {'image': image, 'label': label}

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear layers for queries, keys, and values
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output linear layer
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        seq_len = Q.size(1)

        # Print original shapes of Q, K, V
        print(f"Original Q shape: {Q.shape}")  # Expected: (batch_size, seq_len, d_model)
        print(f"Original K shape: {K.shape}")  # Expected: (batch_size, seq_len, d_model)
        print(f"Original V shape: {V.shape}")  # Expected: (batch_size, seq_len, d_model)

        # Project Q, K, V to the required size (d_model)
        Q = self.W_Q(Q)  # shape: (batch_size, seq_len, d_model)
        K = self.W_K(K)  # shape: (batch_size, seq_len, d_model)
        V = self.W_V(V)  # shape: (batch_size, seq_len, d_model)

        # Print after projection
        print(f"Q after projection: {Q.shape}")  # Expected: (batch_size, seq_len, d_model)
        print(f"K after projection: {K.shape}")  # Expected: (batch_size, seq_len, d_model)
        print(f"V after projection: {V.shape}")  # Expected: (batch_size, seq_len, d_model)

        # Reshape Q, K, V to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Print after reshaping
        print(f"Q after reshape: {Q.shape}")  # Expected: (batch_size, num_heads, seq_len, d_k)
        print(f"K after reshape: {K.shape}")  # Expected: (batch_size, num_heads, seq_len, d_k)
        print(f"V after reshape: {V.shape}")  # Expected: (batch_size, num_heads, seq_len, d_k)

        # Transpose to get dimensions: (batch_size, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)  # Now Q has shape: (batch_size, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)  # Now K has shape: (batch_size, num_heads, seq_len, d_k)
        V = V.transpose(1, 2)  # Now V has shape: (batch_size, num_heads, seq_len, d_k)

        # Print after transposing
        print(f"Q after transpose: {Q.shape}")  # Expected: (batch_size, num_heads, seq_len, d_k)
        print(f"K after transpose: {K.shape}")  # Expected: (batch_size, num_heads, seq_len, d_k)
        print(f"V after transpose: {V.shape}")  # Expected: (batch_size, num_heads, seq_len, d_k)

        # Perform attention mechanism (you can plug in your attention logic here)
        attention_output, attention_weights = self.attention(Q, K, V, mask)

        # Print attention output shape
        print(f"Attention output shape: {attention_output.shape}")  # Expected: (batch_size, seq_len, d_model)

        # Project the output back to the original dimension (d_model)
        attention_output = attention_output.transpose(1, 2).contiguous()  # Transpose back
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)  # Reshape back to (batch_size, seq_len, d_model)

        # Print final output shape
        print(f"Final attention output shape: {attention_output.shape}")  # Expected: (batch_size, seq_len, d_model)

        # Output linear layer
        output = self.W_O(attention_output)  # shape: (batch_size, seq_len, d_model)

        return output, attention_weights


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model  # Dimension of the keys/values

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for Scaled Dot-Product Attention
        Q: Query matrix (batch_size, seq_len, d_model)
        K: Key matrix (batch_size, seq_len, d_model)
        V: Value matrix (batch_size, seq_len, d_model)
        mask: Optional attention mask (batch_size, seq_len)
        """
        # Compute the attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_model ** 0.5  # Scaled dot-product

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Apply mask

        # Apply softmax to normalize the attention scores
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the attention output as a weighted sum of the values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=256, dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention layer
        attention_output, attention_weights = self.multihead_attention(x, x, x, mask)
        
        # Add residual connection and normalize
        x = self.layer_norm1(x + self.dropout(attention_output))

        # Feedforward layer
        feedforward_output = self.feedforward(x)
        
        # Add residual connection and normalize
        x = self.layer_norm2(x + self.dropout(feedforward_output))

        return x, attention_weights


class CombinedModel(nn.Module):
    def __init__(self, model1, model2, input_dim=32, num_heads=8, d_ff=256, dropout=0.3, device='cuda'):
        super(CombinedModel, self).__init__()
        self.device = device

        # Freeze the two models (they are already loaded with pretrained weights)
        for param in model1.parameters():
            param.requires_grad = False
        for param in model2.parameters():
            param.requires_grad = False
        
        # Use only the feature-extracting part of each model
        self.model1 = model1
        self.model2 = model2
        self.attention=None
        # Remove the final layers from each model (regression layer), only use the output features
        self.model1.regression_layer = nn.Identity()
        self.model2.regression_layer = nn.Identity()

        # Define the final regression layer to predict survival (output size 1)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x1, x2):
        # Pass inputs through the two models (each model returns a feature map of size [batch_size, 32])
        features1 = self.model1(x1)  # Shape: (batch_size, 32)
        features2 = self.model2(x2)  # Shape: (batch_size, 32)

        # Debug: Print original feature shapes
        print(f"Original features1 shape: {features1.shape}")  # Should be (batch_size, 32)
        print(f"Original features2 shape: {features2.shape}")  # Should be (batch_size, 32)

        # Concatenate the features along the feature dimension to form a combined feature tensor
        combined_features = torch.cat([features1, features2], dim=1)  # Shape: (batch_size, 64)

        # Debug: Print combined features shape
        print(f"Combined features shape: {combined_features.shape}")  # Should be (batch_size, 64)
        self.attention = TransformerBlock(d_model=combined_features.shape[1])
        # Apply the attention mechanism to the combined features
        # attention_output, _ = self.attention(combined_features, combined_features)

        # Debug: Print attention output shape
       #  print(f"Attention output shape: {attention_output.shape}")  # Should be (batch_size, seq_len, d_model)

        # Attention output has shape (batch_size, seq_len, d_model), so we need to reduce it
        # For example, we can apply global average pooling (if seq_len is 1) or take the first token
        # attention_output = attention_output.mean(dim=1)  # If seq_len > 1, apply mean pooling

        # Debug: Print the final attention output shape before passing to the regression layer
        # print(f"Attention output after pooling shape: {attention_output.shape}")  # Should be (batch_size, d_model)

        # Pass the combined features through the final regression layer
        # survival_output = self.fc(attention_output)  # Shape: (batch_size, 1)
        return combined_features

def save_checkpoint(model, optimizer, epoch, loss, batch=-1, val_loss=None, path='/home/ee577/project/results/Combined_surv.pth', csv_path='/home/ee577/project/results/C_Loss_surv.csv'):
    try:
        # Save checkpoint to the .pth file
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # Save the model's state_dict
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss,
            'val_loss': val_loss,
            'batch': batch,
        }

        # Save the combined model (including attention head) state_dict
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
        combined_model,  # The combined model with attention head
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        num_epochs=1000, 
        patience=10, 
        eval_every=2,  
        max_grad_norm=4.0,
        last_epoch=None
        ):
    combined_model.to(device)
    combined_model.train()

    best_val_loss = float('inf')  # Initialize with a large number
    batches_since_improvement = 0  # Count how many epochs since last improvement
    val_losses = []  # List to store the validation losses
    path = f'/home/ee577/project/results/DTI_surv_checkpoint.pth'
    if last_epoch:
        epoch_range = range(last_epoch, num_epochs)
    else:
        epoch_range = range(num_epochs)

    for epoch in epoch_range:
        random.seed(epoch)
        np.random.seed(epoch)
        torch.manual_seed(epoch)

        # Iterate over batches in the train_loader
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch['image'], batch['label']

            inputs = inputs.to(device)
            labels = labels.to(device)
            transformed_inputs = []

            # Apply random transformations (flip, rotate, noise) to the entire batch
            for i in range(inputs.size(0)):
                image = inputs[i].cpu().numpy()
                if random.random() < 0.5:
                    image = random_flip_rotate(image, epoch)        
                # Ensure image is 4D (1, D, H, W)
                image = torch.tensor(image, dtype=torch.float32).to(device)
                if image.ndimension() == 3:  # (D, H, W) -> Add channel dimension
                    image = image.unsqueeze(0)  # Now it becomes (1, D, H, W)
                transformed_inputs.append(image)

            # Now, stack the images after ensuring all have the shape (1, D, H, W)
            transformed_inputs = torch.stack(transformed_inputs).to(device)

            if combined_model.training:  # Add noise only during training
                noisy_labels = add_noise_to_labels(labels, seed=epoch)  
            optimizer.zero_grad()

            # Forward pass through the two frozen models (DTI and DSC models)
            outputs_1 = DTI_model(transformed_inputs)  # DTI model
            outputs_2 = DSC_model(transformed_inputs)  # DSC model

            # Extract the outputs from both models (excluding last layer)
            features_1 = outputs_1
            features_2 = outputs_2

            # Forward pass through the attention mechanism in the combined model
            # attention_output = combined_model.attention(features_1, features_2)
            attention_output= combined_features = torch.cat([features1, features2], dim=1) # remove later
            # Now, perform the final regression prediction with a new linear layer (survival prediction)
            survival_output = combined_model.fc(attention_output)

            # Smooth L1 Loss for regression
            criterion = nn.SmoothL1Loss()
            train_loss = criterion(survival_output.squeeze(), labels)  # Squeeze to ensure labels match output shape

            l1_loss = combined_model.l1_regularization()

            total_loss = train_loss + l1_loss
            total_loss.backward()
            clip_grad_norm_(combined_model.parameters(), max_grad_norm)
            optimizer.step()

            # Average loss for this epoch
            train_loss = train_loss.item()
            val_loss = evaluate_validation_loss(combined_model, val_loader)  # model in eval mode
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val_loss {val_loss}, Batch: {batch_idx}")

            combined_model.train()  # re-enable gradients for training

            # Perform validation every `eval_every` epochs
            if (batch_idx + 1) % eval_every == 0:
                scheduler.step(val_loss)
                val_losses.append(val_loss)  # Save the validation loss
                save_checkpoint(combined_model, optimizer, epoch=epoch, loss=train_loss, val_loss=val_loss, batch=batch_idx, path=path)
                
                # Early stopping: Check if validation loss has improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    batches_since_improvement = 0
                    # Save model if validation loss improves
                    torch.save(combined_model.state_dict(), f"best_model.pth")
                else:
                    batches_since_improvement += 1

                if batches_since_improvement >= patience:
                    print(f"Early stopping after {epoch + 1} epochs, validation loss has not improved for {patience} epochs.")
                    break

        del inputs, labels, transformed_inputs
        torch.cuda.empty_cache()

batch_siz=16
train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=batch_siz, shuffle=False)
val_loader = DataLoader(CustomDataset(X_val, y_val), batch_size=batch_siz, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test, y_test), batch_size=batch_siz, shuffle=False)
for batch in train_loader:
    survival = batch['image']
    # features = batch['label']
    # print(f" image size: {survival.shape}, label size {features.shape}")

n_shape = batch['image'].shape 
out_shape=y_train.shape[1:][0]

def load_and_freeze_model(model_class, checkpoint_path, model_args=None, freeze_layers=True, remove_last_layer=True, device='cuda'):
    """
    Loads a model checkpoint, freezes layers (optional), and removes the output layer (optional).
    
    Args:
        model_class (nn.Module): The class for the model architecture.
        checkpoint_path (str): Path to the model checkpoint file.
        model_args (dict): Arguments to pass to the model's __init__ method (e.g., {'in_channels': 1, 'out_channels': 55}).
        freeze_layers (bool): Whether to freeze the model layers or not.
        remove_last_layer (bool): Whether to remove the last output layer of the model.
        device (str): Device where the model should be loaded ('cpu' or 'cuda').
        
    Returns:
        nn.Module: The loaded and processed model or None if an error occurred.
    """
    try:
        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize the model with the provided model_args
        model = model_class(**model_args).to(device)  # Pass model_args here

        # Load the model state_dict from the checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print(f"Error: 'model_state_dict' not found in the checkpoint at {checkpoint_path}.")
            return None

        # Freeze the model parameters (except for the last layer if needed)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

        # Optionally remove the last layer (output layer)
        if remove_last_layer:
            if hasattr(model, 'segmentation_head'):
                del model.segmentation_head
            if hasattr(model, 'fc2'):
                del model.fc2

        print(f"Model loaded and prepared from {checkpoint_path}.")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


# Model arguments, ensuring we pass the correct 'in_channels' and 'out_channels'
model_args = {
    'in_channels': 1,  # You can modify this based on your dataset
    'out_channels': out_shape,  # The output channels (e.g., the number of classes for segmentation)
    'device': device,  # The device ('cpu' or 'cuda')
}

# Load the DTI model
DTI_model = load_and_freeze_model(
    model_class=UNet3DRegression_survival,  # Pass the class itself
    checkpoint_path=DTI_mod_path,
    model_args=model_args,  # Pass the model_args dictionary
    freeze_layers=True,
    remove_last_layer=True,
    device=device
)

# Load the DSC model
DSC_model = load_and_freeze_model(
    model_class=UNet3DRegression_survival,
    checkpoint_path=DSC_mod_path,
    model_args=model_args,  # Use the same model_args
    freeze_layers=True,
    remove_last_layer=True,
    device=device
)

# Check if models were loaded successfully
if DTI_model is None:
    print("Failed to load DTI model.")
else:
    print("DTI model loaded successfully.")

if DSC_model is None:
    print("Failed to load DSC model.")
else:
    print("DSC model loaded successfully.")

combined_model = CombinedModel(model1=DSC_model, model2=DTI_model, device=device)

params_to_update = []
for name, param in combined_model.named_parameters():
    if param.requires_grad:  # Only parameters with requires_grad will be updated
        params_to_update.append(param)

optimizer = torch.optim.Adam(params_to_update, lr=1e-4)  # Choose a learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)



train_model(
    combined_model,  # The combined model with attention head
    train_loader, 
    val_loader, 
    optimizer,  # Pass the optimizer
    scheduler,  # Pass the scheduler
    num_epochs=1000, 
    patience=10, 
    eval_every=2,  
    max_grad_norm=4.0
)

