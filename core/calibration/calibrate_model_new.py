import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import psutil
import copy
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from core.calibration.bounds import HB_mu_plus
import pdb
import random

from core.scripts.per_pixel_feature import extract_feature_map

# for conditional conf
from scipy.ndimage import generic_filter, sobel
from scipy.ndimage import filters
from core.condconf import CondConf

import torchvision 
import torchvision.transforms as TVtransforms
import torchvision.transforms.functional as TVtransformsFctl

def get_rcps_losses(model, dataset, rcps_loss_fn, lam, device):
  losses = []
  dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True) 
  for batch in dataloader:
    sets = model.nested_sets_from_output(batch,lam) 
    losses = losses + [rcps_loss_fn(sets, labels),]
  return torch.cat(losses,dim=0)

def get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lam, device):
  losses = []
  dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True) 
  model = model.to(device)
  for batch in dataloader:
    x, labels = batch
    sets = model.nested_sets_from_output(x.to(device),lam) 
    losses = losses + [rcps_loss_fn(sets, labels.to(device)).cpu(),]
  return torch.cat(losses,dim=0)



def get_nonconformity_scores_from_dataset(model,
                                          out_dataset,
                                          device,
                                          lambda_vec,
                                          I, J, num_images,
                                          window_size = 320,
                                          score_defn = "heuristic",
                                          model_type = "UNet"):
    # Create the nonconformity_score_fn with model, device, and lambda_vec
    #nonconformity_score_fn, _ = create_nonconformity_score_fns(model, device, lambda_vec,I, J, num_images,window_size)

    #nonconformity_scores = []  # empty list for nonconformity scores to be appended to

   
    #model = model.to(device)  # load model to GPU
    #print("Printing nonconformity scores for each batch for debugging...")
    # dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    # for batch in dataloader:
    #     x, labels = batch
    #     nonconformity_scores_batch = nonconformity_score_fn(x, labels)
    #     nonconformity_scores.append(torch.tensor(nonconformity_scores_batch, device=device))  # Append as tensor
    # nonconformity_scores = torch.cat(nonconformity_scores, dim=0).detach().cpu().numpy()
    all_nonconformity_scores = []
    
    model = model.to(device)

    for i in range(num_images):
        # Extract the i-th data
        x, labels = out_dataset[i]
        x = x.to(device)
        labels = labels.to(device)

        # Create a new nonconformity score function for each data pair
        nonconformity_score_fn, _ = create_nonconformity_score_fns(model, device, lambda_vec, I, J, 1, window_size,score_defn,model_type)

        # Compute the nonconformity scores for the current image
        nonconformity_scores = nonconformity_score_fn(x, labels)

        # Collect scores
        all_nonconformity_scores.append(nonconformity_scores)

    # Concatenate all nonconformity scores
    nonconformity_scores = np.concatenate(all_nonconformity_scores, axis=0)
      
    return nonconformity_scores

    
def create_nonconformity_score_fns(model,
                                   device, 
                                   lambda_vec, 
                                   I, J, num_images,
                                   window_size=320,
                                   score_defn = "heuristic",
                                   model_type = "UNet"):
    # Given a pretrained model, a device, a lambda grid, and image dimensions (I, J)
    # along with the number of images, returns a score function and its inverse.

    if isinstance(lambda_vec, np.ndarray):
        lambdas = torch.tensor(lambda_vec, dtype=torch.float32).to(device)
    else:
        lambdas = lambda_vec.to(device)

    def nonconformity_score_fn(x, labels):
        # Input: x, labels; both should be 1D vectors of size (num_images * I * J)
        # Output: nonconformity scores as a 1D vector

        # Reshape x and labels if necessary
        if x.ndim == 1:
            x = tensorize(x, num_images, I, J).to(device)
        else:
            x = torch.tensor(x).to(device)

        if labels.ndim == 1:
            labels = torch.tensor(labels.reshape(num_images, I, J)).to(device)
        else:
            labels = torch.tensor(labels).to(device)
        
        all_nonconformity_scores = []
        images_with_inf = []

        for i in range(num_images):
            x_i = x[i].to(device)

            # Add necessary dimensions to x_i if required
            if x_i.ndim == 2:
                x_i = x_i.unsqueeze(0).unsqueeze(0)  # [1, 1, 320, 320]
            elif x_i.ndim == 3:
                if x_i.shape[0] != 1:
                    x_i = x_i.unsqueeze(0)

            label_i = labels[i]

            # Get prediction set components for the image
            heuristic_psets = model(x_i)

            # TODO: if using the trivial model, then fhat
            # will be the blurred ground truth (note that trivialModel nesseciates trivialDataset)
            if model_type == "trivialModel":
              fhat = x # This should be the bluured ground truth
              # print some statistics to check if random_shift is working:
              # Convert tensors to numpy arrays if they aren't already
              fhat_np = fhat.detach().cpu().numpy() if isinstance(fhat, torch.Tensor) else fhat
              x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
              labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
              fhat_summary = np.percentile(fhat_np, [0, 25, 50, 75, 100])
              x_summary = np.percentile(x_np, [0, 25, 50, 75, 100])
              labels_summary = np.percentile(labels_np, [0, 25, 50, 75, 100])
              print("            labels summary:", labels_summary)
              # Print summaries
              print("            fhat summary:", fhat_summary)
              print("            x summary:", x_summary)
              # trivialModel calib and val data will be: (blurred_gt, gt)
            else:
              lb, fhat, ub = heuristic_psets[:, 0, :, :], heuristic_psets[:, 1, :, :], heuristic_psets[:, 2, :, :]
              lhat = fhat + lb
              uhat = ub - fhat

            

            # Compute pixel-wise nonconformity scores

            # Compute pixel-wise nonconformity scores


            if score_defn == "heuristic":
              nonconformity_scores = torch.maximum((fhat - labels)/lhat, (labels - fhat)/uhat)
            elif score_defn == "residual":
              nonconformity_scores = torch.abs((fhat - labels))
            else:
              print(f"###########{model_type}############")
              print(f"###########{score_defn}############")
              raise NotImplementedError("score_defn not implemented")


            #### Old version of heuristic score computation #############
            #nonconformity_scores = torch.full(label_i.shape, float('inf'), dtype=torch.float, device=device)
            # for lam in lambdas:
            #     property_satisfied = containment_check(lam, lhat, fhat, uhat, label_i)
            #     nonconformity_scores = torch.where(
            #         (nonconformity_scores == float('inf')) & property_satisfied,
            #         lam,
            #         nonconformity_scores
            #     )

            if torch.isinf(nonconformity_scores).any():
                images_with_inf.append(i)

            # Crop scores to the center if window_size != 320
            if window_size != 320:
                #nonconformity_scores = get_center_window(nonconformity_scores.squeeze(), window_size)
                nonconformity_scores = vectorize(get_center_window(nonconformity_scores.squeeze(), window_size).detach().cpu().numpy())
            #all_nonconformity_scores.append(vectorize(nonconformity_scores.cpu().numpy())) # .flatten()
            all_nonconformity_scores.append(nonconformity_scores)
        scores = np.concatenate(all_nonconformity_scores)
              #For debugging only
        q1 = np.percentile(scores, 25)
        median = np.percentile(scores, 50)
        q3 = np.percentile(scores, 75)
        max_value = np.max(scores)
        print(f"            Shape of scores: {scores.shape}")
        print(f"            Score distribution: Q1={q1}, Median={median}, Q3={q3}, Max={max_value}")

        return scores

    def nonconformity_score_inv_fn(score_cutoffs, x):
        # Input: score_cutoffs, x; both should be 1D vectors of size (num_images * I * J)
        # Output: the (range of) label pixel intensities such that (pixel-wise), the condition
        # nonconformity_score_fn(x, Output) <= score_cutoffs

        # Reshape x if necessary
        if x.ndim == 1:
            x = tensorize(x, num_images, I, J).to(device)
        else:
            x = torch.tensor(x).to(device)

        if score_cutoffs.ndim == 1:
            score_cutoffs = torch.tensor(score_cutoffs.reshape(num_images, I, J)).to(device)
        else:
            score_cutoffs = torch.tensor(score_cutoffs).to(device)

        all_lower_bounds = []
        all_upper_bounds = []

        for i in range(num_images):
            x_i = x[i].to(device)

            # Add necessary dimensions to x_i if required
            if x_i.ndim == 2:
                x_i = x_i.unsqueeze(0).unsqueeze(0)  # [1, 1, 320, 320]
            elif x_i.ndim == 3:
                if x_i.shape[0] != 1:
                    x_i = x_i.unsqueeze(0)

            score_cutoff_i = score_cutoffs[i]

            # Get the prediction set components for one image
            heuristic_psets = model(x_i)
            if model_type == "trivialModel":
              fhat = x
            else:
              lb, fhat, ub = heuristic_psets[:, 0, :, :], heuristic_psets[:, 1, :, :], heuristic_psets[:, 2, :, :]
              lhat = fhat + lb
              uhat = ub - fhat

            # Calculate the lower and upper bounds for the labels
            if score_defn == "heuristic":
              lower_bound = fhat - score_cutoff_i * lhat
              upper_bound = fhat + score_cutoff_i * uhat
            elif score_defn == "residual":
              lower_bound = fhat - score_cutoff_i
              upper_bound = score_cutoff_i + fhat
            else:
              raise NotImplementedError("score_defn not implemented")

            

            # Crop bounds to the center if window_size != 320
            if window_size != 320:
                lower_bound = get_center_window(lower_bound, window_size)
                upper_bound = get_center_window(upper_bound, window_size)

            #all_lower_bounds.append(lower_bound.flatten().cpu().numpy())
            #all_upper_bounds.append(upper_bound.flatten().cpu().numpy())
            all_lower_bounds.append(vectorize(lower_bound.cpu().numpy())) # make sure we use vectorize() instead of .flatten() for cosistency
            all_upper_bounds.append(vectorize(upper_bound.cpu().numpy()))
        inv_score_lb = np.concatenate(all_lower_bounds)
        inv_score_ub = np.concatenate(all_upper_bounds)

        return [inv_score_ub, inv_score_lb]

    return nonconformity_score_fn, nonconformity_score_inv_fn


def containment_check(lam, lhat, fhat, uhat, labels):
    # Check if label is within the interval [fhat - lambda * lhat, fhat + lambda * uhat]
    lower_bound = fhat - lam * lhat
    upper_bound = fhat + lam * uhat
    return (labels >= lower_bound) & (labels <= upper_bound)




def get_center_window(x, w=100):
    """
    Extracts the center w * w window from the input image.

    Parameters:
    x (torch.Tensor or np.ndarray): The input image of shape (I, J) or (C, I, J) where C is the number of channels.
    w (int): The size of the window (default is 100).

    Returns:
    torch.Tensor or np.ndarray: The center window of size w * w from the input image.
    """
    # Ensure the input is 2D or 3D (grayscale or multi-channel)
    if len(x.shape) not in [2, 3]:
        raise ValueError("Input to get_center_window() must be 2D (I, J) or 3D (C, I, J).")

    # Get the height (I) and width (J) of the image
    if len(x.shape) == 2:
        I, J = x.shape
    else:  # len(x.shape) == 3
        _, I, J = x.shape

    # If the window size is larger than or equal to the image dimensions, return the full image
    if w >= I or w >= J:
        return x

    # Calculate the start and end indices for the center window
    start_i = (I - w) // 2
    start_j = (J - w) // 2
    end_i = start_i + w
    end_j = start_j + w

    # Extract and return the center window
    if len(x.shape) == 2:  # Grayscale image
        return x[start_i:end_i, start_j:end_j]
    else:  # Multi-channel image
        return x[:, start_i:end_i, start_j:end_j]

def generate_const_phi_components(input_dataset, window_size=100):
  """
  Params:
    input_dataset(torch.Tensor): tensor of size (num_images,I,J)
    window_size(int): size of the window of the actual part of each input image used
  
  Returns:
    np.ndarray of size (num_images * window_size * window_size, 1) containing constant 1's
  """

  
  # output: np.ndarray of size (num_images * window_size * window_size, 2)
  # Each row of output will be: (a pixel's intensity in the window of some input image, 
  w = window_size # window size
  #variance_w = w + 2 * edge_nbhd_size # the actual side length needed for variance computation
  
  print(f"            Image  window size: {w}")
  #print(f"            Sliding var est window size: {edge_nbhd_size}")
  num_images = len(input_dataset)
  print(f"            Number of images: {num_images}")
  output_dataset = []
  for i in range(num_images):
    chunk = np.zeros((w**2, 1))
    
    chunk[:,0] = np.ones(w**2)

    # Append the chunk to the output dataset
    output_dataset.append(chunk)

  # Convert the list of chunks to a numpy array
  output_dataset = np.vstack(output_dataset)
  print(f"            Phi data shape: {output_dataset.shape}")
  return output_dataset

def random_shift_input_images(dataset, num_shifts=5, max_shift=0.04,renormalize = False):
    """
    Take all the input images in the grayscale dataset (assumed to be PyTorch tensors), 
    and return a dataset where each input image is the AVERAGE of several randomly shifted
    versions of itself. The result is a single grayscale image for each input that represents 
    the overlap of multiple shifts.
    
    Parameters:
    - dataset: tensor of size (num_images, I, J), assuming this being only downsampled images (i.e. no labels)
    - num_shifts: The number of random shifts to generate and overlap for each image.
    - max_shift: Maximum allowed fraction of height & width of shiftings
    - renormalize: Whether to normalize before and after shifting if there are negative values
    
    Returns:
    - shifted_dataset: The dataset with all input images being the overlap of several shifted versions.
    """
    shifted_images = []

    # Loop through the dataset and apply the random shifts
    for idx in range(len(dataset)):
        image = dataset[idx]  # Assuming each dataset item is (image, label) pair
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
        #input_summary = torch.quantile(image, torch.tensor([0, 0.25, 0.5, 0.75, 1.0]))
        #blurred_summary = torch.quantile(accumulated_image, torch.tensor([0, 0.25, 0.5, 0.75, 1.0]))

        #print(f"Input intensity: {input_summary}")
        #print(f"Re-blurred intensity: {blurred_summary}")
        #print(f"Shifts applied (dx, dy): {shifts}")

        # Add the shifted image and the corresponding label to the new dataset lists
        shifted_images.append(accumulated_image)

    
    # Create a new dataset with the shifted images and original labels
    shifted_dataset =torch.stack([img for img in shifted_images])
    
    return shifted_dataset 



def sobel_operator(image):
  # Construct two ndarrays of same size as the input image
  imx = np.zeros(image.shape)
  imy = np.zeros(image.shape)

  # Run the Sobel operator
  # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html
  filters.sobel(image,1,imx,cval=0.0)  # axis 1 is x
  filters.sobel(image,0,imy, cval=0.0) # axis 0 is y

  return np.sqrt(imx**2+imy**2)



def generate_linear_phi_components_NN(input_dataset,
                                      device,
                                      window_size=100,
                                      d=4):
    """
    Params:
        input_dataset(torch.Tensor): tensor of size (num_images, height, width), assuming this being only downsampled images (i.e. no labels)
        window_size(int): size of the window of the actual part of each input image used
        d: dimension of per-pixel feature map

    Returns:
        np.ndarray of size (num_images * window_size * window_size, d)
    """
    feature_map_model = extract_feature_map(d) # model.forward()
    output_dataset = []
    count = 0
    for img in input_dataset:
      count +=1 
      per_pixel_features, _ = feature_map_model(img.unsqueeze(0))  # Extract per-pixel features
      per_pixel_features = per_pixel_features.detach().cpu()
      per_pixel_features = per_pixel_features.squeeze().numpy()  # Shape: (1 + d, height, width)
      #print(f"      Shape of per_pixel_features from the ({count})th calib img is: {per_pixel_features.shape}")
      w = window_size
      #windowed_per_pixel_features = np.zeros((d+1, w, w))
      chunk = np.zeros((w**2, d+1))
      for i in range(d+1): # Important: note that d+1 since config["d"]=number of non-constant components
        component = per_pixel_features[i]
        #print(f"      Shape of raw condCond NN basis component: {component.shape}")
        vectorized_windowed_component = vectorize(get_center_window(component, w)) # .detach()
        #windowed_per_pixel_features[i] = windowed_component
        chunk[:,i] = vectorized_windowed_component
      output_dataset.append(chunk)
    # Convert the list of chunks to a numpy array
    output_dataset = np.vstack(output_dataset)
    return np.array(output_dataset).reshape((-1,d+1))



def generate_linear_phi_components(input_dataset,
                                  window_size=100,
                                  edge_nbhd_size=20,
                                  basis_components=["const", "xij", "var"]):
    """
    Params:
      input_dataset(torch.Tensor): tensor of size (num_images, I, J), assuming this being only downsampled images (i.e. no labels)
      window_size(int): size of the window of the actual part of each input image used
      edge_nbhd_size(int): window size of edge detection neighborhood needed
      basis_components(list): list specifying which components to include in the output. 
                              Options are: "const", "intensity", "variances"
    
    Returns:
      np.ndarray of size (num_images * window_size * window_size, len(basis_components))
      where each component corresponds to the requested basis components:
        - "const": constant 1's
        - "intensity": pixel intensities
        - "variances": estimated variance of pixels around that pixel
    """



    shifted_dataset = random_shift_input_images(input_dataset)
    w = window_size  # window size for the main window
    variance_w = w + 2 * edge_nbhd_size  # side length needed for variance computation

    print(f"            Image window size: {w}")
    print(f"            Sliding var est window size: {edge_nbhd_size}")
    num_images = len(input_dataset)
    print(f"            Number of images: {num_images}")

    # Number of columns in output based on the length of basis_components
    num_columns = len(basis_components)
    output_dataset = []


    nbhd = edge_nbhd_size
    for i in range(num_images):
        # Initialize the chunk with the correct number of columns based on the components
        chunk = np.zeros((w**2, num_columns))

        # Track which component goes into which column
        current_column = 0

        # "const" component (constant 1's)
        if "const" in basis_components:
            chunk[:, current_column] = np.ones(w**2)
            current_column += 1

        # "intensity" component (pixel intensities)
        if "xij" in basis_components:
            intensity_part = vectorize(get_center_window(input_dataset[i], w))
            chunk[:, current_column] = intensity_part
            current_column += 1

        # "variances" component (estimated variance of pixels in the neighborhood)
        if "var" in basis_components:
            
            variance_window =get_center_window(input_dataset[i], w) # get_center_window(input_dataset[i], variance_w)
            if torch.is_tensor(variance_window):
                variance_window = variance_window.numpy()
            variance_part = vectorize(sobel_operator(variance_window))
            #variance_part = vectorize(get_center_window(generic_filter(variance_window, np.var, size=[nbhd,nbhd]), w))
            chunk[:, current_column] = variance_part
            current_column += 1
          
        if "blurred_xij" in basis_components:
          intensity_part = vectorize(get_center_window(shifted_dataset[i], w))
          chunk[:, current_column] = intensity_part
          current_column += 1
          #(vectorized) blurred xij, blurred through  
        if "blurred_var" in basis_components:
          variance_part = torch.stack([tensorize(variance_part,1,w,w)])
          variance_part = random_shift_input_images(variance_part)[0]
          variance_part = vectorize(variance_part)
          #variance_window = get_center_window(shifted_dataset[i], variance_w)
          # variance_window = get_center_window(shifted_dataset[i],w)
          # if torch.is_tensor(variance_window):
          #   variance_window = variance_window.numpy()
          # variance_part = vectorize(sobel_operator(variance_window))
          # #variance_part = vectorize(get_center_window(generic_filter(variance_window, np.var, size=[nbhd,nbhd]), w))
          chunk[:, current_column] = variance_part
          current_column += 1
         
        # Append the chunk to the output dataset
        output_dataset.append(chunk)

    # Convert the list of chunks to a numpy array
    output_dataset = np.vstack(output_dataset)
    print(f"            Phi data shape: {output_dataset.shape}")
    return output_dataset



# DO NOT DELETE, these are placeholders for condConf
def phi_linear(x):
  """
  input:
    x: result of generate_const_phi_components()
  """
  return x

def phi_const(x):
  """
  x: result of generate_const_phi_components()
  """
  #result = x[:,2]
  return np.ones((x.shape[0],1))


def create_nonconformity_score_fns_modified(model, 
                                            input_dataset,
                                            window_size, 
                                            device, 
                                            lambda_vec,
                                            pixel_idx=None,
                                            score_defn = "heuristic",
                                            model_type = "UNet"): 
                                           
   
    
    # This is a new version of nonconformity score functions to allow more calibration data points
    # by taking more data points but only using the center windows of them 
    # model: trained model
    # input_dataset: a pytorch dataset of (input, output) pairs (this will either be calib set) (or val set)
    # window_size: size of window (as in get_center_window)
    # is_window: is this function creator made for input being image windows or full images; True if for windows
    # I,J: size of the full image
    # num_images: number of data points (e.g. in calib or val set)
    
    # Convert lambda_vec to a PyTorch tensor and move it to the correct device
    if isinstance(lambda_vec, np.ndarray):
        lambdas = torch.tensor(lambda_vec, dtype=torch.float32).to(device)
    else:
        lambdas = lambda_vec.to(device)
  
    def nonconformity_score_fn_modified(x, labels):
      """
      input:
        x: results of generate_linear_phi_components(input_dataset[:,0])
        labels: center windows of label(s) of the corresponding input x's
      """

      w = window_size
      num_images = len(input_dataset)
      print(f"            Number of images for generating score funcs in this dataset: {num_images}")
      
      # step 0: check input sizes
      input_num_images = x.shape[0]/w**2
      input_num_labels = labels.shape[0]/w**2
      if input_num_images != num_images:
        # Raise an exception and print an error message
        raise ValueError(f"            Input size mismatch: Expected {num_images} images, but got {int(input_num_images)} images.")
      if input_num_labels != num_images:
        # Raise an exception and print an error message
        raise ValueError(f"            Input size mismatch: Expected labels for {num_images} images, but got labels for {int(input_num_labels)} images.")


      all_nonconformity_scores = []
      

      for i in range(num_images):
        print(f"            Computing scores for the {i+1}-th image.")

        x_i_full, label_i = input_dataset[i]
        x_i_full = x_i_full.to(device)
        label_i = label_i.to(device)
        #label_i = input_dataset[i][1].to(device)
        #x_i_full = input_dataset[i][0].to(device) # get the ith full image
        # Create a new nonconformity score function for each data pair

        # def create_nonconformity_score_fns(model,
        #                            device, 
        #                            lambda_vec, 
        #                            I, 
        #                            J, 
        #                            num_images,
        #                            window_size=320,
        #                            score_defn = "heuristic",
        #                            model_type = "UNet"):

        nonconformity_score_fn, _ = create_nonconformity_score_fns(model,device, lambda_vec, 320, 320, 1, window_size,score_defn,model_type)

        # Compute the nonconformity scores for the current image
        nonconformity_scores = nonconformity_score_fn(x_i_full, label_i)

        # Collect scores
        all_nonconformity_scores.append(nonconformity_scores)

      # Concatenate all nonconformity scores
      nonconformity_scores = np.concatenate(all_nonconformity_scores, axis=0)

      return nonconformity_scores





      ############## COMMENTED OUT DUE TO TESTING #################
      #   images_with_inf = []
      #   #print(f"                   Shape of x_i_full: {x_i_full.shape}")

      #   # Step 1: get heuristic prediction set components: f, l, u for one image
      #   # Check the number of dimensions and adjust accordingly
      #   if x_i_full.ndim == 2:
      #         # If x_i is 2D (shape [320, 320]), add batch and channel dimensions
      #       x_i_full = x_i_full.unsqueeze(0).unsqueeze(0)  # Result: [1, 1, 320, 320]
      #   elif x_i_full.ndim == 3:
      #         # If x_i is 3D (shape [1, 320, 320] or [320, 320, 1]), add only the batch dimension
      #       if x_i_full.shape[0] != 1:  # If the first dimension is not the channel
      #           x_i_full = x_i_full.unsqueeze(0)  # Result: [1, 1, 320, 320] (assuming first dim is not the channel)
      #       else:
      #           x_i_full = x_i_full.unsqueeze(0)  # Result: [1, 1, 320, 320]
      #   heuristic_psets = model(x_i_full)
      #   lb, fhat, ub = heuristic_psets[:,0,:,:,:], heuristic_psets[:,1,:,:,:], heuristic_psets[:,2,:,:,:]  # lower-hat, f-hat, upper-hat
      #   lhat = fhat + lb
      #   uhat = ub - fhat

      #   # Initialize a tensor to store nonconformity scores for one image,
      #   # starting from inf since nonconformity scores are defined by min...
      #   nonconformity_scores = torch.full(label_i.shape, float('inf'), dtype=torch.float, device=device)

      #   # Step 2: compute pixel-wise nonconformity scores for one image
      #   for lam in lambdas:  # Traverse lambda values from smallest to largest
      #         # Check the property for the current lambda value
      #         property_satisfied = containment_check(lam, lhat, fhat, uhat, label_i)
      #         # Update nonconformity scores where the property is satisfied for the first time
      #         nonconformity_scores = torch.where(
      #           (nonconformity_scores == float('inf')) & property_satisfied,  # only need to look at scores that are still inf
      #           lam,  # if containment condition is satisfied, set that score to lam value
      #           nonconformity_scores  # otherwise keep the original score
      #           )
          
      #   # For debugging only
      #   if torch.isinf(nonconformity_scores).any():
      #     images_with_inf.append(i)
       
       
        
      #   nonconformity_scores = nonconformity_scores.squeeze(0) # squeeze out extra dimension(s)
      #   #print(f"                   Shape of nonconformity_scores: {nonconformity_scores.shape}")
      #   #nonconformity_scores = get_center_window(nonconformity_scores, w)
      #   #all_nonconformity_scores.append(nonconformity_scores.flatten().cpu().numpy())
      #   nonconformity_scores = vectorize(get_center_window(nonconformity_scores, w).cpu().numpy())
      #   all_nonconformity_scores.append(nonconformity_scores)
  
  
      # scores = np.concatenate(all_nonconformity_scores)#.reshape(-1,1)
      
    
      # #For debugging only
      # q1 = np.percentile(scores, 25)
      # median = np.percentile(scores, 50)
      # q3 = np.percentile(scores, 75)
      # max_value = np.max(scores)
      # print(f"            Shape of scores: {scores.shape}")
      # print(f"            Score distribution: Q1={q1}, Median={median}, Q3={q3}, Max={max_value}")
    
      # # Print which images have `inf` values in their scores
      # #if images_with_inf:
      # #  print(f"            Indices of images with inf nonconformity scores: {images_with_inf}")
      # #else:
      # #  print("            No images have inf nonconformity scores.")
 
      # return scores
      ############## COMMENTED OUT DUE TO TESTING #################
      

    def nonconformity_score_inv_fn_modified(score_cutoffs, x):
      w = window_size
      x_full  = input_dataset[0][0].to(device) # this is because no use cases of this inv function requires vectorization (just a single data point)
      # Check the number of dimensions and adjust accordingly
      if x_full.ndim == 2:
          # If x_i is 2D (shape [320, 320]), add batch and channel dimensions
        x_full = x_full.unsqueeze(0).unsqueeze(0)  # Result: [1, 1, 320, 320]
      elif x_full.ndim == 3:
        # If x_i is 3D (shape [1, 320, 320] or [320, 320, 1]), add only the batch dimension
        if x_full.shape[0] != 1:  # If the first dimension is not the channel
          x_full = x_full.unsqueeze(0)  # Result: [1, 1, 320, 320] (assuming first dim is not the channel)
        else:
          x_full = x_full.unsqueeze(0)  # Result: [1, 1, 320, 320]
      heuristic_psets = model(x_full)
      if model_type == "trivialModel":
        fhat = x
      else:
        lb, fhat, ub = heuristic_psets[:,0,:,:,:],heuristic_psets[:,1,:,:,:],heuristic_psets[:,2,:,:,:]
        lhat = (fhat + lb).squeeze(0)
        uhat = (ub - fhat).squeeze(0)
        fhat = fhat.squeeze(0)

      #print(f"            Shape of lhat in inv score: {lhat.shape}")
      if isinstance(score_cutoffs, torch.Tensor):
        score_cutoffs = score_cutoffs.cpu().numpy()

      lhat = vectorize(get_center_window(lhat,w))[pixel_idx]#.numpy()
      fhat = vectorize(get_center_window(fhat,w))[pixel_idx]#.numpy()
      uhat = vectorize(get_center_window(uhat,w))[pixel_idx]#.numpy()


      # Calculate the lower and upper bounds for the labels for one image
      if score_defn == "heuristic":
        lower_bound = fhat - score_cutoffs * lhat
        upper_bound = fhat + score_cutoffs * uhat
      elif score_defn == "residual":
        lower_bound = fhat - score_cutoffs
        upper_bound = score_cutoffs + fhat
      else:
        raise NotImplementedError("score_defn not implemented")
      # lower_bound = fhat - score_cutoffs * lhat
      # upper_bound = fhat + score_cutoffs * uhat
      #B,UB) {lower_bound},{upper_bound}.")
      return [lower_bound,upper_bound]
    
    return nonconformity_score_fn_modified, nonconformity_score_inv_fn_modified


def vectorize(tensor):
    """
    Vectorizes a single tensor so it can be passed to a vectorized basis function.

    Parameters:
    tensor (torch.Tensor): The tensor to be vectorized, 
                           shape (num_images, channels, I, J) or (num_images, I, J).

    Returns:
    vectorized_tensor (np.ndarray): Vectorized tensor as a 1D vector.
    """

    # Ensure the tensor is on CPU and convert to a numpy array
    # tensor_np = tensor.cpu().numpy()

    # Check if the tensor is a PyTorch tensor
    if isinstance(tensor, torch.Tensor):
        # Ensure the tensor is on CPU and convert to a numpy array
        tensor = tensor.detach().cpu().numpy()

    # Vectorize by reshaping it into a 1D vector
    vectorized_tensor = tensor.reshape(-1)

    return vectorized_tensor

def tensorize(vector, num_images,  I=None, J=None):
    """
    Converts a 1D vector back into its original tensor form.

    Parameters:
    vector (np.ndarray): Vectorized tensor of shape (num_images * I * J) or (num_images * I * J * channels).
    num_images (int): The number of images.
    
    I (int): The height of the images.
    J (int): The width of the images.

    Returns:
    tensor (torch.Tensor): The tensorized form of shape (num_images, channels, I, J) or (num_images, I, J).
    """

    tensor = vector.reshape(num_images, I, J)

    # Convert back to a PyTorch tensor
    tensor = torch.tensor(tensor)

    return tensor
  
def calibrate_model_by_CondConf(model, dataset, config):
    # This function calibrates the model using the CondConf framework
    with torch.no_grad():
        print(f"Calibrating...")
        model.eval()
        alpha = config['alpha']
        device = config['device']
        
        print("    Initialize lambda grid")
        if config["uncertainty_type"] == "softmax":
            lambdas = torch.linspace(config['minimum_lambda_softmax'], config['maximum_lambda_softmax'], config['num_lambdas'])
        else:
            lambdas = torch.linspace(config['minimum_lambda'], config['maximum_lambda'], config['num_lambdas'])
        dlambda = lambdas[1] - lambdas[0]  # get the step size (from linspace)
        


        print("    Put trained model on device")
        model = model.to(device)
        model.set_lhat(lambdas[-1] + dlambda - 1e-9)  # set lhat to the maximum value of lambda
        
        print("    Initialize labels")
        if config['dataset'] == 'temca':
            labels = torch.cat([x[1].unsqueeze(0).to('cpu') for x in iter(dataset)], dim=0)
            outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).to('cpu') for x in iter(dataset)])
            print("    Labels initialized.")
        else:
            labels_shape = list(dataset[0][1].unsqueeze(0).shape)#unsequeeze(0) adds a new dimension at the zeroth position
            labels_shape[0] = len(dataset)
            num_images = labels_shape[0] # get the number of image pairs in calibration set
            I,J = dataset[0][1].shape[-2],dataset[0][1].shape[-1]
            print(f"    Height * Width of each image: {I} * {J}")
            image_in_dataset_shape = dataset[0][0].shape
            print(f"    Tensor shape of downsampled image in dataset: {image_in_dataset_shape}")
            print(f"    Number of data points in calibration set: {num_images}")

            labels = torch.zeros(tuple(labels_shape), device='cpu')
            outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
            print(f"    Shape of an input to model() in the original code: {dataset[0][0].unsqueeze(0).shape}")
            outputs_shape[0] = len(dataset)
            outputs = torch.zeros(tuple(outputs_shape), device='cpu')
            print("    Collecting calib set and making predictions for it")
            tempDL = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], pin_memory=True) 
            counter = 0
            for batch in tqdm(tempDL):
                outputs[counter:counter+batch[0].shape[0],:,:,:,:] = model(batch[0].to(device)).cpu()
                # outputs = model prediction, torch.Size([3, 1, 320, 320])
                labels[counter:counter+batch[1].shape[0]] = batch[1]
                # labels = ground truth
                counter += batch[0].shape[0]

        print("    Output: model predictions on calib set & labels")
        out_dataset = TensorDataset(outputs, labels.cpu())

       

        # CondConf framework
        print("    Entered Conditional Conformal")
        print(f"        CondConf device: {device}")
        window_size = config["center_window_size"]
        #variance_nbhd_size = config["var_window_size"]
        print(f"        Using the middle ({window_size}*{window_size}) windows of calib set")
        #print(f"        Variance estimation window size: ({variance_nbhd_size}*{variance_nbhd_size})")
        print( "        Generate calib set for condConf.")
        x_calib = torch.stack([data[0] for data in dataset])
        print(f"            Length of x_calib:{len(x_calib)}")
        #if config['c']
        condConf_basis_type = config['condConf_basis_type'] # possible values: "const", "linear"

        if condConf_basis_type != "linear":
          x_calib = generate_const_phi_components(x_calib,window_size)
        else:
          handcrafted_basis = config["handcrafted_basis_components"]
          if handcrafted_basis != "handcrafted":
            d = config["d"]
            x_calib = x_calib.to(device) # move to gpu for forward pass computing
            #print(f"      Device of x_calib into condConf basis: {x_calib.device}")
            x_calib = generate_linear_phi_components_NN(x_calib, device, window_size,d)
            
          else: 
            basis_components = config["basis_components"]
            x_calib = generate_linear_phi_components(x_calib, window_size, variance_nbhd_size,basis_components)

        y_calib = np.concatenate([vectorize(get_center_window(dataset[i][1],window_size)) for i in range(len(dataset))])


        
        infinite_params = {}
        if condConf_basis_type == "const":
          phi_fn = phi_const #phi_fn_const
        elif condConf_basis_type == "linear":
          phi_fn = phi_linear #create_phi_fn_linear(I, J, num_images)
        else:
          raise ValueError(f"Invalid value for condConf basis func type: {condConf_basis_type}.")
        print(f"        Finite-dimensional basis (Phi) is {condConf_basis_type}")

        print( "        Create nonconformity score function and its inverse")
        #nonconformity_score_fn, nonconformity_score_inv_fn = create_nonconformity_score_fns(model, device, lambdas,I, J, num_images)
        score_defn = config['score_defn']
        model_type = config['model']
        # 
# def create_nonconformity_score_fns_modified(model, 
#                                             input_dataset,
#                                             window_size, 
#                                             device, 
#                                             lambda_vec,
#                                             pixel_idx=None,
#                                             score_defn = "heuristic",
#                                             model_type = "UNet"): 
        nonconformity_score_fn_modified, _ = create_nonconformity_score_fns_modified(model, 
                                                                                    dataset,
                                                                                    window_size,  
                                                                                    device,  
                                                                                    lambdas,
                                                                                    None,
                                                                                    score_defn,
                                                                                    model_type)

        print( "        Set up the condConf optimization problem")
        cond_conf = CondConf(score_fn  = nonconformity_score_fn_modified,
                             Phi_fn = phi_fn,
                             quantile_fn= None, 
                             infinite_params = infinite_params,
                             seed=0)
        print(f"        Shapes of x_calib(with phi components) and y_calib: {[x_calib.shape, y_calib.shape]}")
        cond_conf.setup_problem(x_calib, y_calib)   


        



        print("    Applying split CP for calib loss table")
        # Retrieve the dataloader and get nonconformity scores
        nonconformity_scores = get_nonconformity_scores_from_dataset(model,
                                                                    dataset,
                                                                    device,
                                                                    lambdas,
                                                                    I,
                                                                    J,
                                                                    num_images,
                                                                    window_size,
                                                                    score_defn,
                                                                    model_type)
        
        # Set the value of lam to be the 1 - (alpha / T) quantile of the nonconformity scores
        print("        Pick the adjusted conformal quantile")
        lam = torch.quantile(torch.tensor(nonconformity_scores), 1 - alpha / num_images)
        model.set_lhat(lam)  # set the lam value of model to the tuned lam value
        print("")
        print(f"            Model's lambdahat set to {model.lhat} by split CP")
        # Compute the loss table for each lambda value, 
        # starting from largest in grid and stopping at tunned lambda value

        # # For debugging condConf only:
        # print(f"            Validating the two score functions")
        # noncomformity_scores_from_modified_score_func = nonconformity_score_fn_modified(x_calib, y_calib)
        # diff = nonconformity_scores - noncomformity_scores_from_modified_score_func
        # print("            Original Scores    |    Modified Scores    |    Difference")
        # print("            ---------------------------------------------------------")
        # for orig, mod, d in zip(nonconformity_scores, noncomformity_scores_from_modified_score_func, diff):
        #   print(f"            {orig:<20} | {mod:<20} | {d:<20}")
        # loss table based on CP prediction set
        print("            Computing calib loss table")
        calib_loss_table = torch.zeros((outputs.shape[0], lambdas.shape[0]), device='cpu')
        rcps_loss_fn = get_rcps_loss_fn(config)
        for lamd_value in reversed(lambdas):
            losses = get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lamd_value, device)
            calib_loss_table[:, np.where(lambdas==lamd_value)[0]] = losses[:, None]
            if lamd_value == lam:
                break
    print("    DONE!")
    return model, calib_loss_table, cond_conf









def get_rcps_metrics_from_outputs(model, out_dataset, rcps_loss_fn, device):
  losses = []
  sizes = []
  residuals = []
  spatial_miscoverages = []
  dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True) 
  model = model.to(device)
  for batch in dataloader:
    x, labels = batch
    labels = labels.to(device)
    sets = model.nested_sets_from_output(x.to(device)) 
    losses = losses + [rcps_loss_fn(sets, labels),]
    sets_full = (sets[2]-sets[0]).flatten(start_dim=1).detach().cpu().numpy()
    size_random_idxs = np.random.choice(sets_full.shape[1],size=sets_full.shape[0])
    size_samples = sets_full[range(sets_full.shape[0]),size_random_idxs]
    residuals = residuals + [(labels - sets[1]).abs().flatten(start_dim=1)[range(sets_full.shape[0]),size_random_idxs]]
    spatial_miscoverages = spatial_miscoverages + [(labels > sets[2]).float() + (labels < sets[0]).float()]
    sizes = sizes + [torch.tensor(size_samples),]
  losses = torch.cat(losses,dim=0)
  sizes = torch.cat(sizes,dim=0)
  sizes = sizes + torch.rand(size=sizes.shape).to(sizes.device)*1e-6
  residuals = torch.cat(residuals,dim=0).detach().cpu().numpy() 
  spearman = spearmanr(residuals, sizes)[0]
  mse = (residuals*residuals).mean().item()
  spatial_miscoverage = torch.cat(spatial_miscoverages, dim=0).detach().cpu().numpy().mean(axis=0).mean(axis=0)
  size_bins = torch.tensor([0, torch.quantile(sizes, 0.25), torch.quantile(sizes, 0.5), torch.quantile(sizes, 0.75)])
  buckets = torch.bucketize(sizes, size_bins)-1
  stratified_risks = torch.tensor([losses[buckets == bucket].mean() for bucket in range(size_bins.shape[0])])
  print(f"Model output shape: {x.shape}, label shape: {labels.shape}, Sets shape: {sets[2].shape}, sizes: {sizes}, size_bins:{size_bins}, stratified_risks: {stratified_risks}, mse: {mse}")
  return losses, sizes, spearman, stratified_risks, mse, spatial_miscoverage

def evaluate_from_loss_table(loss_table,n,alpha,delta):
  with torch.no_grad():
    perm = torch.randperm(loss_table.shape[0])
    loss_table = loss_table[perm]
    calib_table, val_table = loss_table[:n], loss_table[n:]
    Rhats = calib_table.mean(dim=0)
    RhatPlus = torch.tensor([HB_mu_plus(Rhat, n, delta) for Rhat in Rhats])
    try:
        idx_lambda = (RhatPlus <= delta).nonzero()[0]
    except:
        print("No rejections made!")
        idx_lambda = 0
    return val_table[:,idx_lambda].mean()
  
def fraction_missed_loss(pset,label): # prediction set and label
  misses = (pset[0].squeeze() > label.squeeze()).float() + (pset[2].squeeze() < label.squeeze()).float()
  misses[misses > 1.0] = 1.0
  d = len(misses.shape)
  return misses.mean(dim=tuple(range(1,d)))




def get_rcps_loss_fn(config):
  string = config['rcps_loss']
  if string == 'fraction_missed':
    return fraction_missed_loss
  else:
    raise NotImplementedError

def calibrate_model(model, dataset, config):
  # This function takes a trained model (including heuristic upper & lower lengths)
  # and a calibration dataset to calibrate the model by tunning a lambdahat value 
  # to generate CRPS.

  with torch.no_grad():
    print(f"Calibrating...")
    model.eval()
    alpha = config['alpha']
    delta = config['delta']
    device = config['device']
    print("Initialize lambdas")
    if config["uncertainty_type"] == "softmax":
      lambdas = torch.linspace(config['minimum_lambda_softmax'],config['maximum_lambda_softmax'],config['num_lambdas'])
    else:
      lambdas = torch.linspace(config['minimum_lambda'],config['maximum_lambda'],config['num_lambdas'])
    print("Initialize loss")
    rcps_loss_fn = get_rcps_loss_fn(config) # speicify what loss function (L; for prediction set) you use to find lambda (e.g. fraction of missed pixels)
    print("Put model on device")
    model = model.to(device)
    print("Initialize labels")
    if config['dataset'] == 'temca':
      labels = torch.cat([x[1].unsqueeze(0).to('cpu') for x in iter(dataset)], dim=0)
      outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).to('cpu') for x in iter(dataset)])
      print("Labels initialized.")
    else:
      labels_shape = list(dataset[0][1].unsqueeze(0).shape)
      labels_shape[0] = len(dataset)
      labels = torch.zeros(tuple(labels_shape), device='cpu')
      print(f"Size of an input to model() in the original code: {dataset[0][0].unsqueeze(0).shape}")
      outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
      outputs_shape[0] = len(dataset)
      outputs = torch.zeros(tuple(outputs_shape),device='cpu')
      print("Collecting dataset")
      tempDL = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], pin_memory=True) 
      counter = 0
      for batch in tqdm(tempDL):
        outputs[counter:counter+batch[0].shape[0],:,:,:,:] = model(batch[0].to(device)).cpu()
        labels[counter:counter+batch[1].shape[0]] = batch[1]
        counter += batch[0].shape[0]


    print("Output dataset")
    out_dataset = TensorDataset(outputs,labels.cpu())
    dlambda = lambdas[1]-lambdas[0] # get the step size (from linspace)
    model.set_lhat(lambdas[-1]+dlambda-1e-9) # set lhat to the maximum value of lambda
    print("Computing losses")
    calib_loss_table = torch.zeros((outputs.shape[0],lambdas.shape[0]))
    for lam in reversed(lambdas): # from largest to smallest
      losses = get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lam-dlambda, device)
        # losses = a vector of losses (=missed fractions) on each calibration image when inflator = lam-dlambda
      calib_loss_table[:,np.where(lambdas==lam)[0]] = losses[:,None]
        # fill in loss table (to keep track of the lambdahat tunning procedure)
      Rhat = losses.mean()
        # average loss, Rhat
      RhatPlus = HB_mu_plus(Rhat.item(), losses.shape[0], delta)
        # modified avg loss using concentration bound (e.g. hoeffding), = R hat + other stuff
      print(f"\rLambda: {lam:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}  ",end='')
      if Rhat >= alpha or RhatPlus > alpha:
        model.set_lhat(lam) # the "lhat" in "set_lhat" really means "lambda hat", not to be confused with "lower hat"
        print("")
        print(f"Model's lhat set to {model.lhat}")
        break
    return model, calib_loss_table


def calibrate_model_by_CP(model, dataset, config):
    # This function calibrates the model using the CP framework
    with torch.no_grad():
        print(f"Calibrating...")
        model.eval()
        alpha = config['alpha']
        device = config['device']
        print("Initialize lambda grid")
        if config["uncertainty_type"] == "softmax":
            lambdas = torch.linspace(config['minimum_lambda_softmax'], config['maximum_lambda_softmax'], config['num_lambdas'])
        else:
            lambdas = torch.linspace(config['minimum_lambda'], config['maximum_lambda'], config['num_lambdas'])

        print("Put model on device")
        model = model.to(device)
        
        print("Initialize labels")
        if config['dataset'] == 'temca':
            labels = torch.cat([x[1].unsqueeze(0).to('cpu') for x in iter(dataset)], dim=0)
            outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).to('cpu') for x in iter(dataset)])
            print("Labels initialized.")
        else:
            labels_shape = list(dataset[0][1].unsqueeze(0).shape)
            labels_shape[0] = len(dataset)
            T_value = labels_shape[0] # get the number of image pairs in calibration set
            num_images = labels_shape[0] # get the number of image pairs in calibration set
            I,J = dataset[0][1].shape[-2],dataset[0][1].shape[-1]
            print(f"Size of each image: {I} * {J}")
            print("")
            print(f"Number of data points in calibration set: {num_images}")
            labels = torch.zeros(tuple(labels_shape), device='cpu')
            outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
            outputs_shape[0] = len(dataset)
            outputs = torch.zeros(tuple(outputs_shape), device='cpu')
            print("Collecting dataset")
            tempDL = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], pin_memory=True) 
          
            counter = 0
            for batch in tqdm(tempDL):
                outputs[counter:counter+batch[0].shape[0],:,:,:,:] = model(batch[0].to(device)).cpu()
                labels[counter:counter+batch[1].shape[0]] = batch[1]
                counter += batch[0].shape[0]

        print("Output dataset")
        out_dataset = TensorDataset(outputs, labels.cpu())

        dlambda = lambdas[1] - lambdas[0]  # get the step size (from linspace)
        model.set_lhat(lambdas[-1] + dlambda - 1e-9)  # set lhat to the maximum value of lambda
        print("Computing nonconformity scores of calibration set")
        score_defn = config['score_defn']
        model_type = config['model']
        # Retrieve the dataloader and get nonconformity scores
        nonconformity_scores = get_nonconformity_scores_from_dataset(model, dataset, device, lambdas,I, J, num_images,score_defn,model_type)
        
        # Set the value of lam to be the 1 - (alpha / T) quantile of the nonconformity scores
        print("Picking the adjusted conformal quantile")
        lam = torch.quantile(torch.tensor(nonconformity_scores), 1 - alpha / T_value)
        model.set_lhat(lam)  # set the lam value of model to the tuned lam value
        print("")
        print(f"Model's lambdahat is set to {model.lhat}")

        # Compute the loss table for each lambda value, 
        # starting from largest in grid and stopping at tunned lambda value
        print("Computing loss table")
        calib_loss_table = torch.zeros((outputs.shape[0], lambdas.shape[0]), device='cpu')
        rcps_loss_fn = get_rcps_loss_fn(config)
        for lamd_value in reversed(lambdas):
            losses = get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lamd_value, device)
            calib_loss_table[:, np.where(lambdas==lamd_value)[0]] = losses[:, None]
            if lamd_value == lam:
                break

    print("DONE!")
    return model, calib_loss_table












