import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
# import wandb
import random
import pickle
import copy
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import pickle as pkl
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as TVtransforms
import torchvision.transforms.functional as TVtransformsFctl
import warnings
import yaml

#from core.scripts.train import train_net

from core.models.add_uncertainty import add_uncertainty
from core.utils import fix_randomness 
from core.datasets.utils import normalize_dataset 



# Models
from core.models.trunks.unet import UNet


# Dataset
from tqdm import tqdm
from core.datasets.fastmri import FastMRIDataset



# Step 1: obtain training data (X_i, |Y_i-f(X_i)|) where f() is the pre-trained U-Net

# Step 2: Train a neural network (e.g. fully convoluted) on the training data
    # This NN should learn a d-dimensional per-pixel feature map (i.e. a d-dimensional feature vector at each pixel location)
    # of the input image where d is customizable

# Step 3: Extract the d-dimensional feature map as a callable function, the structure of this
# function should be similar to the following:
# def generate_linear_phi_components(input_dataset,
#                                   window_size=100,
#                                   d):
#     """
#     Params:
#       input_dataset(torch.Tensor): tensor of size (num_images, I, J), assuming this being only downsampled images (i.e. no labels)
#       window_size(int): size of the window of the actual part of each input image used
#       d: dimension of per-pixel feature map
    
#     Returns:
#       np.ndarray of size (num_images * window_size * window_size, d)
#     """









def get_NNbasis_config():
  config = {
    "program": "core/scripts/router.py",
    "method": "grid",
    "metric": {
        "goal": "minimize",
        "name": "mean_size"
    },
    "name": "fastmri_test",
    "project": "fastmri_test",
    "group": "fastmri_test",
    "dataset": "fastmri",
    "num_inputs":1,
    "model": "UNet" ,
    "uncertainty_type": "quantiles",
    "alpha":0.1,
    "delta": 0.1, # only for RCPS
    "num_lambdas": 2000,
    "rcps_loss":  "fraction_missed",
    "minimum_lambda_softmax": 0,
    "maximum_lambda_softmax":  1.2,
    "minimum_lambda": 0,
    "maximum_lambda": 20,
    "device": "cuda:0",
    "epochs": 10,
    "batch_size": 16, # for UNet training
    "lr": 0.001, # 0.001, 0.0001 # learning rate
    "load_from_checkpoint":  True,
    "checkpoint_dir": "experiments/fastmri_test/checkpoints",
    "checkpoint_every": 1,
    "validate_every": 10,
    "num_validation_images":10, # for final image saving in eval.py
    "q_lo": 0.05,
    "q_hi": 0.95,
    "q_lo_weight":  1,
    "q_hi_weight": 1,
    "mse_weight": 1,
    "num_softmax": 50,
    "input_normalization": "standard",
    "output_normalization":  "min-max"
    }
  return config


def generate_dataset(fraction = 0.03):
    
    # fraction = how much of the original FastMRI dataset to use to train the per-pixel map

    
    # load some parameters
    config = get_NNbasis_config()
    # if os.path.exists(results_fname):
    #     print(f"Results already precomputed and stored in {results_fname}!")
    #     os._exit(os.EX_OK) 
    # else:
    #     print("Computing the results from scratch!")
    # # Otherwise compute results
    curr_method = config["uncertainty_type"]
    curr_lr = config["lr"]
    curr_dataset = config["dataset"]

    params = {key: config[key] for key in config.keys() }
    batch_size = config['batch_size']
    params['batch_size'] = batch_size

    


    # # save residual dataset
    # dataset_path = '/project2/rina/lekunbillwang/im2im-uq/core/datasets/bill_test_fastmri' 
    #     #'/project2/rina/lekunbillwang/im2im-uq/core/datasets/bill_test_fastmri/singlecoil_val'
    # dataset_save_path = os.path.join(dataset_path, f"residual_dataset_{fraction}.pkl")
    # if os.path.exists(dataset_save_path):
    #     print(f"            Loading precomputed residual dataset from {dataset_save_path}...")
    #     with open(dataset_save_path, "rb") as f:
    #         residual_dataset = pickle.load(f)
    #     return residual_dataset
    # else:
    #     print("         No residual dataset found. Generating residual data from scratch...")




    # load FastMRI dataset 
    dataset_path = '/project2/rina/lekunbillwang/im2im-uq/core/datasets/bill_test_fastmri' 
    dataset_path = os.path.join(dataset_path, 'singlecoil_val')
    mask_info = {'type': 'equispaced', 'center_fraction': [0.08], 'acceleration': [4]}
    dataset = FastMRIDataset(dataset_path, normalize_input=config['input_normalization'], normalize_output=config['output_normalization'], mask_info=mask_info)
    dataset = normalize_dataset(dataset)

    frac_unused_data = 1-fraction
    print(f"            Using only {fraction} of FastMRI for basis training")
    lengths = np.round(len(dataset) * np.array([fraction, frac_unused_data])).astype(int)
    lengths[-1] = len(dataset) - (lengths.sum() - lengths[-1])
    dataset, _ = torch.utils.data.random_split(dataset, lengths.tolist()) 


    # train/load U-Net model
    frac_train_data = 1
    frac_val_data = 1-frac_train_data
    lengths = np.round(len(dataset) * np.array([frac_train_data, frac_val_data])).astype(int)
    lengths[-1] = len(dataset) - (lengths.sum() - lengths[-1])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths.tolist()) 
    
    trunk = UNet(config["num_inputs"],1)
    model = add_uncertainty(trunk, params)
    
    
    # Model Load Method 1
    #from core.scripts.train_mri_offline import train_net
    # model = train_net(model,
    #                 train_dataset,
    #                 val_dataset,
    #                 config['device'],
    #                 config['epochs'],
    #                 batch_size,
    #                 config['lr'],
    #                 config['load_from_checkpoint'],
    #                 config['checkpoint_dir'],
    #                 config['checkpoint_every'],
    #                 config['validate_every'],
    #                 params)  
    
    
    # Model Load Method 2
    checkpoint_final_path = (
    config['checkpoint_dir'] +
    f"/CP_epoch{config['epochs']}_" +
    config['dataset'] + "_" +
    config['uncertainty_type'] + "_" +
    str(config['batch_size']) + "_" +
    str(config['lr']) + "_" +
    config['input_normalization'] + "_" +
    config['output_normalization'].replace('.', '_') +'.pth')
    

    # Fixed syntax errors in the code block
    if os.path.exists(checkpoint_final_path):
        # try:
        net = torch.load(checkpoint_final_path)
        net.eval()
        print(f"Model loaded from checkpoint {checkpoint_final_path}")
        model = net

    




    device = config['device']
    labels_shape = list(dataset[0][1].unsqueeze(0).shape)
    labels_shape[0] = len(dataset)

    inputs = torch.zeros(tuple(labels_shape), device='cpu')

    residuals = torch.zeros(tuple(labels_shape), device='cpu')

    print("         Creating residual dataset")
    tempDL = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], pin_memory=True) 
    counter = 0
    model.eval() # TODO: maybe I can avoid adding the uncertainty and save even more space?
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(tempDL):
            
            noisy_inputs = batch[0].cpu()
            #print(f"            Noisy inputs shape: {noisy_inputs.shape}")
            inputs[counter:counter+batch[1].shape[0]] = noisy_inputs
            
            ground_truths = batch[1].cpu()
            #print(f"            Ground truths shape: {ground_truths.shape}")
            
            outputs = model(batch[0].to(device))
            #print(f"            UNet original outputs shape: {outputs.shape}")
            
            predictions = outputs[:,1,:,:,:].cpu()
            #print(f"            UNet predictions shape: {predictions.shape}")
            
            residuals[counter:counter+batch[0].shape[0]] = torch.abs(predictions - ground_truths)
            
            counter += batch[0].shape[0]
    
    residual_dataset = [(inp, residual) for inp, residual in zip(inputs, residuals)]
    # with open(dataset_save_path, "wb") as f:
    #     pickle.dump(residual_dataset, f)
    return residual_dataset



# define a Convolutional Auto Encoder for feature-map learning
class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels, feature_dim):
        """
        Convolutional Autoencoder with fixed components in latent space.

        Params:
            input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            feature_dim (int): Number of learned components in the latent space (excluding any fixed components (currently only a constant component)).
        """
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: Downsampling the input and compressing to (2 + feature_dim) channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),  # d-dimensional learned features
            nn.ReLU()
        )
        
        # Decoder: Reconstructing the input from the compressed feature map
        self.decoder = nn.Sequential(
            nn.Conv2d(1 + feature_dim, 64, kernel_size=3, padding=1), # "1+" because of constant part
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),  # Reconstruct original channels
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        # Compute learned latent features
        learned_features = self.encoder(x)  # Shape: (batch_size, feature_dim, height, width)
        #print(f"            Shape of learned_features:{learned_features.shape}")
        
        # Construct the full latent space by adding fixed components
        #print("Inputs device to forward():", x.device)
        ones = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, requires_grad=False)  # Constant part
        latent = torch.cat([ones, learned_features], dim=1)  # Concatenate along channel dimension
    
        # Debugging: Ensure latent requires grad
        #print("Learned features require grad:", learned_features.requires_grad)
        #print("Latent requires grad:", latent.requires_grad)  # This should be True

        # Reconstruct the input
        reconstruction = self.decoder(latent)

        # Debugging: Ensure reconstruction requires grad
        # print("Reconstruction requires grad:", reconstruction.requires_grad)  # This should be True
    
        return latent, reconstruction




def train_autoencoder(d, batch_size=16, epochs=10, lr=0.001):
    """
    Train a convolutional autoencoder and save progress after each epoch. 
    If a checkpoint exists, resume training from the latest checkpoint.

    Params:
        d (int): Dimensionality of the latent per-pixel feature map.
        batch_size (int): Batch size for training.
        epochs (int): Total number of epochs to train.
        lr (float): Learning rate.

    Returns:
        nn.Module: The trained autoencoder model.
    """
    torch.set_grad_enabled(True) # enable gradient for basis training
    #print("Grad mode enabled:", torch.is_grad_enabled())
    config = get_NNbasis_config()
    device = config['device']
    
    #TODO: check that if model has been loaded, sip the dataset loading and directly
    # outputting the model
    # Ensure the save directory exists
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Check for existing checkpoints and if so just load trained model
    checkpoint_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("autoencoder_epoch_")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])  # Extract epoch number
    )
    
    
    if checkpoint_files:
        latest_checkpoint = os.path.join(save_dir, checkpoint_files[-1])
        print(f"            Found a trained model. Loading from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        input_channels = checkpoint["model_state_dict"]["encoder.0.weight"].shape[1]
        model = ConvAutoencoder(input_channels, d)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        print("         Trained model for condConf basis loaded.")
        model = model.to(device)
        return model
        
    return("            No trained odel for condConf basis found. Train from scratch!")
    dataset = generate_dataset()  # Load or generate dataset

    # Initialize the model
    #print(f"Shape of input image to CAE: {dataset[0][0].shape}")
    input_channels = dataset[0][0].shape[0] # the shape of first 
    #print(f"Num of input channels to CAE: {input_channels}")
    model = ConvAutoencoder(input_channels, d)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Reconstruction loss

    
    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = model.to(device)
    
    start_epoch = 0
    print("CondConf Basis NN training starts!")
    if checkpoint_files:
        latest_checkpoint = os.path.join(save_dir, checkpoint_files[-1])
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    
    
        
    for epoch in range(start_epoch, epochs):
        print(f"            CondConf Basis Training Epoch: {epoch}")
        model.train()
        train_loss = 0
        for inputs, ground_truths in train_loader:  # Assuming DataLoader returns (inputs, labels)
            for param in model.parameters():
                assert param.requires_grad, "Model parameter does not require grad."
                param.requires_grad = True
            inputs, ground_truths = inputs.to(device), ground_truths.to(device)
            #print(f"            Shape of inputs from loader: {inputs.shape}")
            #print(f"            Shape of inputs from loader: {ground_truths.shape}")
            optimizer.zero_grad()

            # Forward pass
            latent, reconstruction = model(inputs)
            #print("Latent requires grad:", latent.requires_grad)
            #print("Reconstruction requires grad:", reconstruction.requires_grad)


            # Compute loss
            loss = criterion(reconstruction, ground_truths)
             # Check loss gradient
            # print("Loss requires grad:", loss.requires_grad)
            if not loss.requires_grad:
                raise RuntimeError("Loss tensor does not require grad. Check computation graph.")
            loss.backward()
           
            optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save the model checkpoint after each epoch
        checkpoint_path = os.path.join(save_dir, f"autoencoder_epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    return model






def extract_feature_map(d): # this will be called in calibrate_model_new.py
    """
    Extract per-pixel feature maps using the trained encoder of the AE.
    
    Params:
        model (nn.Module): Trained convolutional autoencoder.
        dataset (torch.utils.data.Dataset): Input dataset.
        device (str): Device ('cuda' or 'cpu').
        batch_size (int): Batch size for processing.
    """

    config = get_NNbasis_config()
    device = config['device']
    model = train_autoencoder(d)
    model = model.to(device)
    model.eval()
    
    return model.forward








