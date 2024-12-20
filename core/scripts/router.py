import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
#os.environ["WANDB_MODE"]="offline"
#os.environ["WANDB__SERVICE_WAIT"] = "300"
import wandb
import random
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
from core.scripts.train_mri_offline import train_net
from core.scripts.eval import get_images,  get_images_condConf_parallel, eval_net, get_loss_table, eval_set_metrics ,get_images_condConf_non_parallel
from core.models.add_uncertainty import add_uncertainty
from core.calibration.calibrate_model_new import calibrate_model, calibrate_model_by_CP, calibrate_model_by_CondConf
from core.utils import fix_randomness 
from core.datasets.utils import normalize_dataset 

# Models
from core.models.trunks.unet import UNet

# Datasets
from core.datasets.bsbcm import BSBCMDataset
from core.datasets.fastmri import FastMRIDataset
from core.datasets.temca import TEMCADataset


# condConf
from scipy.ndimage import generic_filter
from core.condconf import CondConf


def get_config():
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
    #"output_dir": "experiments/fastmri_train/outputs/raw",
    "output_dir": "experiments/fastmri_test/outputs/raw",
    "dataset": "fastmri",
    "num_inputs":1,
    "data_split_percentages": [0.8, 0.1, 0.1, 0.0], # model_tr, calib, val(plot), unused
    "model": "UNet", # "UNet" "trivialModel"
        # Note that trivialModel means the base prediction model will return the blurred ground truth,
        # this is implemented by directly modifying the score functions instead of 
        # defining a trivial constant model, and this must be used along with a trivialDataset
        # where each pair of data is (blurred ground truth, groud truth)
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
    "output_normalization":  "min-max",

    # Beblow are parameters for calibration modifications:
    "UQ_method": "Cond CP" , # "Risk Control", "Cond CP"
    "score_defn": "residual", # "heuristic"
    "condConf_testing_mode": True,
    "testing_by_shifting": "no_shift", # "no_shift" # whether input imgaes on calib and val sets have been randomly shifted
    "num_calib_img": 5, # 30
    "num_val_img": 2, # 5
    "condConf_basis_type": "linear",# "linear" # const
    "handcrafted_basis_components": "NN_basis", #"handcrafted"
    "center_window_size": 40 ,# side length of center window for calibration set
    "val_center_window_size": 50 # 200 # side length of center window for validation set
  }
  if config["handcrafted_basis_components"] == "handcrafted":
    config["basis_components"] = ["const","xij","var","blurred_xij","blurred_var"]
    config["var_window_size"]= 5 # side length of sliding variance window
  if config["handcrafted_basis_components"] == "NN_basis":
    config["d"]=3
  return config

def get_img_generating_fname(config):
    results_fname = (f"experiments/fastmri_test/outputs/raw/results_{config['dataset']}_"
                     f"{config['model']}_"
                     f"{config['uncertainty_type']}_"
                     f"{config['batch_size']}_"
                     f"{config['lr']}_"
                     f"{config['input_normalization']}_"
                     f"({config['output_normalization'].replace('.', '_')})_"
                     f"({config['num_calib_img']})_"
                     f"({config['num_val_img']})_"
                     f"{config['center_window_size']}_"
                     f"{config['val_center_window_size']}_"
                     f"{config['condConf_basis_type']}_"
                     f"{config['score_defn']}_"
                     f"{config['testing_by_shifting']}_"
                     f"({config['handcrafted_basis_components']})_" #f"({','.join(config['handcrafted_basis_components'])})_"
                     f"{config['alpha']}.pkl")  # Modified alpha format
    return results_fname

def get_img_save_fname(config):
    results_fname = (f"experiments/fastmri_test/outputs/images/{config['dataset']}_"
                     f"{config['model']}_"
                     f"({config['num_calib_img']})_"
                     f"({config['num_val_img']})_"
                     f"{config['center_window_size']}_"
                     f"{config['val_center_window_size']}_"
                     f"{config['condConf_basis_type']}_"
                     f"{config['score_defn']}_"
                     f"{config['testing_by_shifting']}_"
                     f"({config['handcrafted_basis_components']})_"#f"({','.join(config['handcrafted_basis_components'])})_"
                     f"{config['alpha']}")  # Modified alpha format
    return results_fname



def normalize_01(x):
  x = x - x.min()
  x = x / x.max()
  return x

def random_shift_input_images(dataset, num_shifts=3, max_shift=0.2, renormalize=False):
    """
    Take all the input images in the grayscale dataset (assumed to be PyTorch tensors), 
    and return a dataset where each input image is the AVERAGE of several randomly shifted
    versions of itself. The result is a single grayscale image for each input that represents 
    the overlap of multiple shifts.
    
    Parameters:
    - dataset: A tuple or list-like structure, where dataset[0] contains input grayscale images as tensors.
    - num_shifts: The number of random shifts to generate and overlap for each image.
    - max_shift: Maximum allowed fraction of height & width of shiftings
    - renormalize: Whether to normalize before and after shifting if there are negative values
    
    Returns:
    - shifted_dataset: The dataset with all input images being the overlap of several shifted versions.
    """
    shifted_images = []
    labels = []

    # Loop through the dataset and apply the random shifts
    for idx in range(len(dataset)):
        image, label = dataset[idx]  # Assuming each dataset item is (image, label) pair
        print(image.shape)
        
        # Ensure the image has the right shape: (channels, height, width)
        if len(image.shape) == 2:  # If it's a 2D grayscale image, add a channel dimension
            image = image.unsqueeze(0) # image.shape = (nchanel, height, width)

        # Calculate the original min and max values for renormalization
        original_min, original_max = image.min(), image.max()
        
        # Check if there are any negative values in the image
        if renormalize and torch.any(image < 0):
            print("Normalizing to (0,1) before random shifting")
            image = normalize_01(image)

        # Convert tensor to PIL image (ensure the input is in the correct format)
        image_pil = TVtransforms.ToPILImage()(image)

        # Deep copy for accumulation
        # accumulated_image = image.clone()  
        # accumulated_image = TVtransforms.ToPILImage()(accumulated_image)
        # accumulated_image = TVtransforms.ToTensor()(accumulated_image)


        # Calculate the minimum pixel intensity for padding
        min_intensity = float(image.min().item())

        accumulated_image = torch.zeros_like(image)
        for idx in range(num_shifts):
            # Apply RandomAffine with the minimum intensity as padding
            transform = TVtransforms.RandomAffine(degrees=0, translate=(max_shift, max_shift)) #, fill=min_intensity
            shifted_image = transform(image_pil)
            shifted_image = TVtransforms.ToTensor()(shifted_image) + 1e-6
            if idx == 0:
              accumulated_image = shifted_image
            else:
              accumulated_image += shifted_image

        # Divide by the number of random shifts performed 
        accumulated_image /= num_shifts # (num_shifts + 1)

        # Rescale accumulated image back to the original (min, max) range
        if renormalize and torch.any(image < 0):
            print("Normalizing back to original scale after random shifting")
            accumulated_image = accumulated_image * (original_max - original_min) + original_min

        # Add the shifted image and the corresponding label to the new dataset lists
        shifted_images.append(accumulated_image)
        labels.append(label)

    # Create a new dataset with the shifted images and original labels
    shifted_dataset = [(image, label) for image, label in zip(shifted_images, labels)]
    
    return shifted_dataset



def random_shift_input_images_by_padding(dataset, num_shifts=5, max_shift=0.01,renormalize = False):
    """
    Take all the input images in the grayscale dataset (assumed to be PyTorch tensors), 
    and return a dataset where each input image is the AVERAGE of several randomly shifted
    versions of itself. The result is a single grayscale image for each input that represents 
    the overlap of multiple shifts.
    
    Parameters:
    - dataset: A tuple or list-like structure, where dataset[0] contains input grayscale images as tensors.
    - num_shifts: The number of random shifts to generate and overlap for each image.
    - max_shift: Maximum allowed fraction of height & width of shiftings
    - renormalize: Whether to normalize before and after shifting if there are negative values
    
    Returns:
    - shifted_dataset: The dataset with all input images being the overlap of several shifted versions.
    """
    shifted_images = []
    labels = []

    # Loop through the dataset and apply the random shifts
    for idx in range(len(dataset)):
        image, label = dataset[idx]  # Assuming each dataset item is (image, label) pair
        print(image.shape)
        
        # Ensure the image has the right shape: (channels, height, width)
        if len(image.shape) == 2:  # If it's a 2D grayscale image, add a channel dimension
            image = image.unsqueeze(0) # image.shape = (nchanel, height, width)
        _, height, width = image.shape

        # Calculate the original min and max values for renormalization
        original_min, original_max = image.min(), image.max()

        # Calculate the minimum pixel intensity for padding
        min_intensity = float(image.min().item())

        accumulated_image = torch.zeros_like(image)
        shifts = [] 
        for idx in range(num_shifts):
            # Calculate maximum pixel shifts
            max_dx = int(max_shift * width)
            max_dy = int(max_shift * height)

            # Randomly choose shift amounts
            dx = random.randint(-max_dx, max_dx) # dx > 0 = shift to the right by dx; dx < 0 = shift to left by |dx|
            dy = random.randint(-max_dy, max_dy) # dy > 0 = shift up
            shifts.append((dx, dy))  # Track shift applied
            print((dx,dy))

            # Shift image by padding and slicing
            padding = (max(dx, 0), max(-dx, 0), max(-dy, 0), max(dy, 0))
            #print(padding)
            shifted_image = torch.nn.functional.pad(image, padding, mode="constant", value=min_intensity)
            #print(shifted_image.shape)

            # Crop to original dimensions after padding
            shifted_image = shifted_image[:, max(dy, 0):(height + max(dy, 0)), max(-dx, 0):(width + max(-dx, 0))]
            #print(shifted_image.shape)
            if idx == 0:
              accumulated_image = shifted_image
            else:
              accumulated_image += shifted_image

        # Divide by the number of random shifts performed 
        accumulated_image /= num_shifts # (num_shifts + 1)
         # Print the five-point summaries for input and blurred images
        input_summary = torch.quantile(image, torch.tensor([0, 0.25, 0.5, 0.75, 1.0]))
        blurred_summary = torch.quantile(accumulated_image, torch.tensor([0, 0.25, 0.5, 0.75, 1.0]))

        print(f"Input intensity: {input_summary}")
        print(f"Re-blurred intensity: {blurred_summary}")
        print(f"Shifts applied (dx, dy): {shifts}")

        # Add the shifted image and the corresponding label to the new dataset lists
        shifted_images.append(accumulated_image)
        labels.append(label)

    # Create a new dataset with the shifted images and original labels
    shifted_dataset = [(image, label) for image, label in zip(shifted_images, labels)]
    
    return shifted_dataset    


# Define a wrapper to replace inputs with ground truth images
class TrivialDataset(torch.utils.data.Dataset):
  # dataset = TrivialDataset(dataset) will substitute all the input images in dataset with 
  # the corresponding ground truth
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        input_image, ground_truth = self.original_dataset[idx]
        # Replace the input image with the ground truth image
        return ground_truth, ground_truth




if __name__ == "__main__":
  # Fix the randomness
  fix_randomness()
  warnings.filterwarnings("ignore")

  print("Entered main method.")
  config = get_config()
  wandb.init(mode="offline",config=config) 
  print("wandb init.")
  # Check if results exist already
  output_dir = config['output_dir'] 
  #results_fname = output_dir + f'/results_' + wandb.config['dataset'] + "_" + wandb.config['uncertainty_type'] + "_" + str(wandb.config['batch_size']) + "_" + str(wandb.config['lr']) +"_" + wandb.config['input_normalization'] + "_" + wandb.config['output_normalization'].replace('.','_') + '.pkl'
  # results_fname = (f"{output_dir}/results_{wandb.config['dataset']}_"
  #                f"{wandb.config['uncertainty_type']}_"
  #                f"{wandb.config['batch_size']}_"
  #                f"{wandb.config['lr']}_"
  #                f"{wandb.config['input_normalization']}_"
  #                f"({wandb.config['output_normalization'].replace('.', '_')})_"
  #                f"({wandb.config['num_calib_img']})_"
  #                f"({wandb.config['num_val_img']})_"
  #                f"{wandb.config['center_window_size']}_"
  #                f"{wandb.config['val_center_window_size']}_"
  #                f"{wandb.config['condConf_basis_type']}.pkl") 

  results_fname = (f"{get_img_generating_fname(config)}")
  print(results_fname)
  

  #results_fname = get_img_generating_fname(wandb.config)

  if os.path.exists(results_fname):
    print(f"Results already precomputed and stored in {results_fname}!")
    os._exit(os.EX_OK) 
  else:
    print("Computing the results from scratch!")
  # Otherwise compute results
  curr_method = wandb.config["uncertainty_type"]
  curr_lr = wandb.config["lr"]
  curr_dataset = wandb.config["dataset"]
  wandb.run.name = f"{curr_method}, {curr_dataset}, lr{curr_lr}"
  wandb.run.save()
  params = { key: wandb.config[key] for key in wandb.config.keys() }
  batch_size = wandb.config['batch_size']
  params['batch_size'] = batch_size
  print("wandb save run.")

  # DATASET LOADING
  if wandb.config["dataset"] == "CIFAR10":
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = T.Compose([ T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize ])
    dataset = torchvision.datasets.CIFAR10('/clusterfs/abc/angelopoulos/CIFAR10', download=True, transform=transform)
  elif wandb.config["dataset"] == "bsbcm":
    path = '/home/aa/data/bsbcm'
    dataset = BSBCMDataset(path, num_instances='all', normalize=wandb.config["output_normalization"])
  elif wandb.config["dataset"] == "fastmri":
    path = '/project2/rina/lekunbillwang/im2im-uq/core/datasets/bill_test_fastmri/singlecoil_val'#singlecoil_train' # singlecoil_val #NOTE: singlecoil_val won't work as it does not have labels
    mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
    dataset = FastMRIDataset(path, normalize_input=wandb.config["input_normalization"], normalize_output = wandb.config["output_normalization"], mask_info=mask_info)
    dataset = normalize_dataset(dataset)
    wandb.config.update(dataset.norm_params)
    params.update(dataset.norm_params)
  elif wandb.config["dataset"] == "temca":
    path = '/clusterfs/fiona/amit/temca_data/'
    dataset = TEMCADataset(path, patch_size=[wandb.config["side_length"], wandb.config["side_length"]], downsampling=[wandb.config["downsampling_factor"],wandb.config["downsampling_factor"]], num_imgs='all', buffer_size=wandb.config["num_buffer"], normalize='01') 
  else:
    raise NotImplementedError 

  # MODEL LOADING
  if wandb.config["dataset"] == "CIFAR10":
    if wandb.config["model"] == "ResNet18":
      trunk = torchvision.models.resnet18(num_classes=wandb.config["num_classes"])
  if (wandb.config["model"] == "UNet") or (wandb.config["model"] == "trivialModel"): 
    trunk = UNet(wandb.config["num_inputs"],1)
  
  # ADD LAST LAYER OF MODEL
  model = add_uncertainty(trunk, params)

  # DATA SPLITTING
  if wandb.config["dataset"] == "temca":
    img_paths = dataset.img_paths
    lengths = np.round(len(img_paths)*np.array(wandb.config["data_split_percentages"])).astype(int)
    lengths[-1] = len(img_paths)-(lengths.sum()-lengths[-1])
    random.shuffle(img_paths)
    train_dataset = copy.deepcopy(dataset)
    calib_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)
    train_dataset.img_paths = img_paths[:lengths[0]]
    calib_dataset.img_paths = img_paths[lengths[0]:(lengths[0]+lengths[1])]
    val_dataset.img_paths = img_paths[(lengths[0]+lengths[1]):(lengths[0]+lengths[1]+lengths[2])]
  else:
    print(f"Total number of data points in the dataset is {len(dataset)}")
    lengths = np.round(len(dataset)*np.array(wandb.config["data_split_percentages"])).astype(int)
    lengths[-1] = len(dataset)-(lengths.sum()-lengths[-1])
    train_dataset, calib_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, lengths.tolist()) 
    
    num_calib_length = wandb.config["num_calib_img"]
    num_val_img =  wandb.config["num_val_img"]
    if wandb.config["condConf_testing_mode"]:
       print("condConf testing mode is ON.")
       
       if len(calib_dataset) > num_calib_length:
        print(f"    Use (at most) {num_calib_length} data points for calib.")
        calib_indices = np.random.choice(len(calib_dataset), size=num_calib_length, replace=False)
        calib_dataset = torch.utils.data.Subset(calib_dataset, calib_indices)
        #print(f"    Length of calib set: {len(calib_dataset)}.")

       if len(val_dataset) > num_val_img:
        print(f"    Use (at most) {num_val_img} data points for val & image generations.")
        val_indices = np.random.choice(len(val_dataset), size=num_val_img, replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
    if wandb.config["model"] == "trivialModel":
      print(f"    Using trivial base model")
      calib_dataset = TrivialDataset(calib_dataset)
      val_dataset = TrivialDataset(val_dataset)
    if config['testing_by_shifting'] == "shift":
      print(f"    Applying random shift to input images")
      calib_dataset = random_shift_input_images_by_padding(calib_dataset) # random_shift_input_images()
      val_dataset = random_shift_input_images_by_padding(val_dataset)  # random_shift_input_images()


  model = train_net(model,
                    train_dataset,
                    val_dataset,
                    wandb.config['device'],
                    wandb.config['epochs'],
                    batch_size,
                    wandb.config['lr'],
                    wandb.config['load_from_checkpoint'],
                    wandb.config['checkpoint_dir'],
                    wandb.config['checkpoint_every'],
                    wandb.config['validate_every'],
                    params)   

  print("Done training!")
  model.eval() # turn to evaluation mode
  with torch.no_grad():
    print("Get the validation loss table.") # Doing this first, so I can save it for later experiments.
    val_loss_table = get_loss_table(model,val_dataset,wandb.config)
    print("Calibrate the model's lambdahat/Set up condConf optimization problem")
    calib_method = wandb.config["UQ_method"]
    print(f"The calibration method is: {calib_method}")
    if calib_method == "Risk Control":
      model, calib_loss_table = calibrate_model(model, calib_dataset, params)
      print(f"Model calibrated by RCPS! lambda hat = {model.lhat}")
    elif calib_method == "CP":
      model, calib_loss_table = calibrate_model_by_CP(model, calib_dataset, params)
      print(f"Model calibrated by split CP! lambda hat = {model.lhat}")
    elif calib_method == "Cond CP":
      model, calib_loss_table, condConf = calibrate_model_by_CondConf(model, calib_dataset, params)
      print(f"The condConf optimization problem is set up!")
    
    
    # Save the loss tables
    if output_dir != None:
        try:
            os.makedirs(output_dir,exist_ok=True)
            print('Created output directory')
        except OSError:
            pass
    calib_loss_table = calib_loss_table.to("cpu")
    val_loss_table = val_loss_table.to("cpu")
    combined_loss_table = torch.cat((calib_loss_table, val_loss_table), dim=0)
    combined_loss_table = combined_loss_table.to('cpu')
    output_path = output_dir + f'/loss_table_{wandb.config["dataset"]}_{wandb.config["uncertainty_type"]}_{wandb.config["batch_size"]}_{wandb.config["lr"]}_{wandb.config["input_normalization"]}_{wandb.config["output_normalization"].replace(".", "_")}.pth'
    torch.save(combined_loss_table, output_path)
    #torch.save(torch.cat((calib_loss_table,val_loss_table),dim=0),output_dir + f'/loss_table_' + wandb.config['dataset'] + "_" + wandb.config['uncertainty_type'] + "_" + str(wandb.config['batch_size']) + "_" + str(wandb.config['lr']) + "_" + wandb.config['input_normalization'] + "_" + wandb.config['output_normalization'].replace('.','_') + '.pth')
    print("Loss table for calib and val sets saved!")






    # Get the prediction sets and properly organize them 
    print("Get prediction sets, images, and metrics on validation set")

    if (calib_method == "Risk Control") or (calib_method == "CP"):
      examples_input, examples_lower_edge, examples_prediction, examples_upper_edge, examples_ground_truth, examples_ll, examples_ul, raw_images_dict = get_images(model,
                                                                                                                                                                  val_dataset,
                                                                                                                                                                  wandb.config['device'],
                                                                                                                                                                  list(range(wandb.config['num_val_img'])),
                                                                                                                                                                  params)
    elif calib_method == "Cond CP":
      if config["uncertainty_type"] == "softmax":
        lambdas = torch.linspace(wandb.config['minimum_lambda_softmax'], wandb.config['maximum_lambda_softmax'], config['num_lambdas'])
      else:
        lambdas = torch.linspace(wandb.config['minimum_lambda'], wandb.config['maximum_lambda'], wandb.config['num_lambdas'])
      #raw_images_dict = get_images_condConf_parallel(model,val_dataset,wandb.config['device'],list(range(wandb.config['num_val_img'])),params,lambdas,condConf)
      raw_images_dict = get_images_condConf_non_parallel(model,val_dataset,wandb.config['device'],list(range(wandb.config['num_val_img'])),params,lambdas,condConf)
      CP_raw_image_dict = get_images(model,
                                     val_dataset,
                                     wandb.config['device'],
                                     list(range(wandb.config['num_val_img'])),
                                     params)

    
    # Log everything
    #wandb.log({"epoch": wandb.config['epochs']+1, "examples_input": examples_input})
    #wandb.log({"epoch": wandb.config['epochs']+1, "Lower edge": examples_lower_edge})
    #wandb.log({"epoch": wandb.config['epochs']+1, "Predictions": examples_prediction})
    #wandb.log({"epoch": wandb.config['epochs']+1, "Upper edge": examples_upper_edge})
    #wandb.log({"epoch": wandb.config['epochs']+1, "Ground truth": examples_ground_truth})
    #wandb.log({"epoch": wandb.config['epochs']+1, "Lower length": examples_ll})
    #wandb.log({"epoch": wandb.config['epochs']+1, "Upper length": examples_ul})

    
    # Get the risk and other metrics 
    print("GET THE METRICS INCLUDING SPATIAL MISCOVERAGE")
    risk, sizes, spearman, stratified_risk, mse, spatial_miscoverage = eval_set_metrics(model, val_dataset, params)
    print("DONE")


    #data = [[label, val] for (label, val) in zip(["Easy","Easy-medium", "Medium-Hard", "Hard"], stratified_risk.numpy())]
    #table = wandb.Table(data=data, columns = ["Difficulty", "Empirical Risk"])
    #wandb.log({"Size-Stratified Risk Barplot" : wandb.plot.bar(table, "Difficulty","Empirical Risk", title="Size-Stratified Risk") })

    print(f"Risk: {risk}  |  Mean size: {sizes.mean()}  |  Spearman: {spearman}  |  Size-stratified risk: {stratified_risk} | MSE: {mse} | Spatial miscoverage: (mu, sigma, min, max) = ({spatial_miscoverage.mean()}, {spatial_miscoverage.std()}, {spatial_miscoverage.min()}, {spatial_miscoverage.max()})")
    wandb.log({"epoch": wandb.config['epochs']+1, "risk": risk, "mean_size":sizes.mean(), "Spearman":spearman, "Size-Stratified Risk":stratified_risk, "mse":mse, "spatial_miscoverage" : spatial_miscoverage})
    
    # Save outputs for later plotting
    print("Saving outputs for plotting")
    if output_dir != None:
        try:
            os.makedirs(output_dir,exist_ok=True)
            print('Created output directory')
        except OSError:
            pass
        results = { "risk": risk, "sizes": sizes, "spearman": spearman, "size-stratified risk": stratified_risk, "mse": mse, "spatial_miscoverage": spatial_miscoverage }
        results.update(raw_images_dict)
        results.update(CP_raw_image_dict)
        with open(results_fname, 'wb') as handle:
          pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

        print(f'Results saved to file {results_fname}!')

    print(f"Done with {str(params)}")
