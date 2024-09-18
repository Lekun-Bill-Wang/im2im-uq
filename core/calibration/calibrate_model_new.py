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

# for conditional conf
from scipy.ndimage import generic_filter
from core.condconf import CondConf

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




def phi_fn_const(x):
  # If x is a PyTorch tensor, convert to NumPy array
  if torch.is_tensor(x):
    if x.is_cuda:
        x = x.cpu()
    x = x.numpy()
  results = np.ones((x.shape[0], 1))
  print(f"Shape of the constant basis: {results.shape}")
  return results



def create_phi_fn_linear(I, J, num_images):
    """
    Function creator that returns a function to process images as described.

    Parameters:
    I, J (int): Dimensions of each image.
    num_images (int): The number of images.

    Returns:
    function: A function that takes a 1D vector of size I * J * num_images and
              returns phi value for each element (pixel) in the input
    """

    def phi_fn_linear(x):
        """
        x is a 1D vector of size I * J * num_images,
        where each chunk of size I * J corresponds to a vectorized image.

        Returns phi value for each element (pixel) in the input
        """
        # If x is a PyTorch tensor, convert to NumPy array
        if torch.is_tensor(x):
            if x.is_cuda:
                x = x.cpu()
            x = x.numpy()

        # Reshape x into individual vectorized images
        images = x.reshape(num_images, I, J)

        # Initialize an empty list to store the results for each image
        results = []

        for img in images:
            # First dimension: a sheet of ones
            ones_sheet = np.ones((I, J))
            # Second dimension: variance in a 10x10 neighborhood around each pixel
            variance_sheet = generic_filter(img, np.var, size=10)
            # Third dimension: original pixel intensities
            intensity_sheet = img.copy()
            # Stack these three sheets along a new first dimension
            result = np.stack([ones_sheet, variance_sheet, intensity_sheet], axis=0)
            # Append the result for the current image to the results list
            results.append(result)

        print(f"Shape of the linear basis: {results.shape}")
        return results #final_result

    return phi_fn_linear


# This will be kept for loss table computations
def create_nonconformity_score_fns(model, device, lambda_vec, I, J, num_images):
    # Given a pretrained model, a device, a lambda grid, and image dimensions (I, J)
    # along with the number of images, returns a score function and its inverse.
    
    # Convert lambda_vec to a PyTorch tensor and move it to the correct device

    if isinstance(lambda_vec, np.ndarray):
        lambdas = torch.tensor(lambda_vec, dtype=torch.float32).to(device)
    else:
        lambdas = lambda_vec.to(device)
  
    def nonconformity_score_fn(x, labels):
    # Has access to: model, device, lambdas, I, J, num_images
    # Input: x, labels; both should be 1D vectors of size (num_images * I * J)
    # Output: nonconformity scores as a 1D vector
    
      # Check if x and labels need to be reshaped (this function allows x to be in a 1-way or 3-way tensor)
      if x.ndim == 1:
          x = x = tensorize(x,num_images, I,J).to(device)
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

          # Check the number of dimensions and adjust accordingly
          if x_i.ndim == 2:
              # If x_i is 2D (shape [320, 320]), add batch and channel dimensions
              x_i = x_i.unsqueeze(0).unsqueeze(0)  # Result: [1, 1, 320, 320]
          elif x_i.ndim == 3:
              # If x_i is 3D (shape [1, 320, 320] or [320, 320, 1]), add only the batch dimension
              if x_i.shape[0] != 1:  # If the first dimension is not the channel
                  x_i = x_i.unsqueeze(0)  # Result: [1, 1, 320, 320] (assuming first dim is not the channel)
              else:
                  x_i = x_i.unsqueeze(0)  # Result: [1, 1, 320, 320]

          label_i = labels[i]

          # Step 1: get heuristic prediction set components: f, l, u for one image
          heuristic_psets = model(x_i)
          lb, fhat, ub = heuristic_psets[:,0,:,:,:], heuristic_psets[:,1,:,:,:], heuristic_psets[:,2,:,:,:]  # lower-hat, f-hat, upper-hat
          lhat = fhat + lb
          uhat = ub - fhat

          # Initialize a tensor to store nonconformity scores for one image,
          # starting from inf since nonconformity scores are defined by min...
          nonconformity_scores = torch.full(label_i.shape, float('inf'), dtype=torch.float, device=device)

          # Step 2: compute pixel-wise nonconformity scores for one image
          for lam in lambdas:  # Traverse lambda values from smallest to largest
              # Check the property for the current lambda value
              property_satisfied = containment_check(lam, lhat, fhat, uhat, label_i)
              # Update nonconformity scores where the property is satisfied for the first time
              nonconformity_scores = torch.where(
                (nonconformity_scores == float('inf')) & property_satisfied,  # only need to look at scores that are still inf
                lam,  # if containment condition is satisfied, set that score to lam value
                nonconformity_scores  # otherwise keep the original score
                )
          
          # Check if there are any `inf` values in the scores for this image
          if torch.isinf(nonconformity_scores).any():
              images_with_inf.append(i)

          all_nonconformity_scores.append(nonconformity_scores.flatten().cpu().numpy())
        
      # Flatten the scores back into a 1D vector for all images
      scores = np.concatenate(all_nonconformity_scores)
    
      # Calculate the quartiles using NumPy
      q1 = np.percentile(scores, 25)
      median = np.percentile(scores, 50)
      q3 = np.percentile(scores, 75)
      max_value = np.max(scores)
    
      # Print the summary
      #print(f"Score distribution: Q1={q1}, Median={median}, Q3={q3}, Max={max_value}")
    
      # Print which images have `inf` values in their scores
      #if images_with_inf:
      #  print(f"Indices of images with inf nonconformity scores: {images_with_inf}")
      #else:
      #  print("No images have inf nonconformity scores.")
 
      return scores

      

    def nonconformity_score_inv_fn(score_cutoffs, x):
        # Has access to: model, device, lambdas, I, J, num_images
        # Input: score_cutoffs, x; both should be 1D vectors of size (num_images * I * J)
        # Output: the (range of) label pixel intensities such that (pixel-wise), the condition
        # nonconformity_score_fn(x, Output) <= score_cutoffs

        # Check if x needs to be reshaped
        if x.ndim == 1:
            x = tensorize(x,num_images, I,J).to(device)
        else:
            x = torch.tensor(x).to(device)
        
        if score_cutoffs.ndim == 1:
            score_cutoffs = torch.tensor(score_cutoffs.reshape(num_images, I, J)).to(device)
        else:
            score_cutoffs = torch.tensor(score_cutoffs).to(device)

        all_lower_bounds = []
        all_upper_bounds = []

        for i in range(num_images):
            print(f"            Inverting the score for the {i}-th image")
            x_i = x[i].to(device)

            # Check the number of dimensions and adjust accordingly
            if x_i.ndim == 2:
              # If x_i is 2D (shape [320, 320]), add batch and channel dimensions
              x_i = x_i.unsqueeze(0).unsqueeze(0)  # Result: [1, 1, 320, 320]
            elif x_i.ndim == 3:
              # If x_i is 3D (shape [1, 320, 320] or [320, 320, 1]), add only the batch dimension
              if x_i.shape[0] != 1:  # If the first dimension is not the channel
                x_i = x_i.unsqueeze(0)  # Result: [1, 1, 320, 320] (assuming first dim is not the channel)
              else:
                x_i = x_i.unsqueeze(0)  # Result: [1, 1, 320, 320]
    
    
            score_cutoff_i = score_cutoffs[i]

            # Get the prediction set components for one image
            heuristic_psets = model(x_i)
            lb, fhat, ub = heuristic_psets[:,0,:,:,:],heuristic_psets[:,1,:,:,:],heuristic_psets[:,2,:,:,:]
            lhat = fhat + lb
            uhat = ub - fhat

            # Calculate the lower and upper bounds for the labels for one image
            lower_bound = fhat - score_cutoff_i * lhat
            upper_bound = fhat + score_cutoff_i * uhat

            all_lower_bounds.append(lower_bound.flatten().cpu().numpy())
            all_upper_bounds.append(upper_bound.flatten().cpu().numpy())

        ub = np.concatenate(all_upper_bounds)
        lb = np.concatenate(all_lower_bounds)
        print(f"Sizes of nonconf scores inverted: {[len(ub),len(lb)]}")
        # Flatten the bounds back into 1D vectors for all images
        return [ub,lb]
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


def generate_linear_phi_components(input_dataset, window_size=100, edge_nbhd_size = 20):
  """
  Params:
    input_dataset(torch.Tensor): tensor of size (num_images,I,J)
    window_size(int): size of the window of the actual part of each input image used
    edge_nbhd_size: integer, window size of edge detection nbhd needed
  
  Returns:
    np.ndarray of size (num_images * window_size * window_size, 3)
    the first component: pixel intensities
    the second component: estimated variance of pixels around that pixel
    the thrid component: constant 1's
  """

  
  # output: np.ndarray of size (num_images * window_size * window_size, 2)
  # Each row of output will be: (a pixel's intensity in the window of some input image, 
  w = window_size # window size
  variance_w = w + 2 * edge_nbhd_size # the actual side length needed for variance computation
  
  print(f"            Image  window size: {w}")
  print(f"            Sliding var est window size: {edge_nbhd_size}")
  num_images = len(input_dataset)
  print(f"            Number of images: {num_images}")
  output_dataset = []
  for i in range(num_images):
    chunk = np.zeros((w**2, 3))
    #print(f"        Chunk size: {w}")
    # get variance
    variance_window = get_center_window(input_dataset[i],variance_w)
    if torch.is_tensor(variance_window):
      variance_window = variance_window.numpy()
    variance_part = vectorize(get_center_window(generic_filter(variance_window, np.var, size=edge_nbhd_size),w))
    # get intensity
    intensity_part = vectorize(get_center_window(input_dataset[i],w))
    # combine
    chunk[:,0] = intensity_part
    chunk[:,1] = variance_part
    chunk[:,2] = np.ones(w**2)
    # Append the chunk to the output dataset
    output_dataset.append(chunk)

  # Convert the list of chunks to a numpy array
  output_dataset = np.vstack(output_dataset)
  print(f"            Phi data shape: {output_dataset.shape}")
  return output_dataset


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
                                            pixel_idx=None): # is_window, I,J,num_images
    # TODO: modify according to phi_components()
    # Given a pretrained model, a device, a lambda grid, and image dimensions (I, J)
    # along with the number of images, returns a score function and its inverse.
    
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
    # Has access to: model, input_dataset, window_size, device, lambda_vec, I, J, num_images

    
      
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

      
      # if x.ndim == 1: # if x is already vectorized, then we need to reshape it to sheets of images...
      #   if is_window: # ...according to whether the input x is from windows or full images
      #     x =  tensorize(x,num_images, w,w).to(device)
      #   else:
      #     x =  tensorize(x,num_images, I,J).to(device)
      # else:
      #     x = torch.tensor(x).to(device)

          
    
      # if labels.ndim == 1:
      #     labels = torch.tensor(labels.reshape(num_images, I, J)).to(device) # 
      # else:
      #     labels = torch.tensor(labels).to(device)

      all_nonconformity_scores = []
      images_with_inf = []

      for i in range(num_images):
        print(f"            Computing scores for the {i+1}-th image.")
        label_i = input_dataset[i][1].to(device)
        x_i_full = input_dataset[i][0].to(device) # get the ith full image
        
        #print(f"                   Shape of x_i_full: {x_i_full.shape}")

        # Step 1: get heuristic prediction set components: f, l, u for one image
        # Check the number of dimensions and adjust accordingly
        if x_i_full.ndim == 2:
              # If x_i is 2D (shape [320, 320]), add batch and channel dimensions
            x_i_full = x_i_full.unsqueeze(0).unsqueeze(0)  # Result: [1, 1, 320, 320]
        elif x_i_full.ndim == 3:
              # If x_i is 3D (shape [1, 320, 320] or [320, 320, 1]), add only the batch dimension
            if x_i_full.shape[0] != 1:  # If the first dimension is not the channel
                x_i_full = x_i_full.unsqueeze(0)  # Result: [1, 1, 320, 320] (assuming first dim is not the channel)
            else:
                x_i_full = x_i_full.unsqueeze(0)  # Result: [1, 1, 320, 320]
        heuristic_psets = model(x_i_full)
        lb, fhat, ub = heuristic_psets[:,0,:,:,:], heuristic_psets[:,1,:,:,:], heuristic_psets[:,2,:,:,:]  # lower-hat, f-hat, upper-hat
        lhat = fhat + lb
        uhat = ub - fhat

        # Initialize a tensor to store nonconformity scores for one image,
        # starting from inf since nonconformity scores are defined by min...
        nonconformity_scores = torch.full(label_i.shape, float('inf'), dtype=torch.float, device=device)

        # Step 2: compute pixel-wise nonconformity scores for one image
        for lam in lambdas:  # Traverse lambda values from smallest to largest
              # Check the property for the current lambda value
              property_satisfied = containment_check(lam, lhat, fhat, uhat, label_i)
              # Update nonconformity scores where the property is satisfied for the first time
              nonconformity_scores = torch.where(
                (nonconformity_scores == float('inf')) & property_satisfied,  # only need to look at scores that are still inf
                lam,  # if containment condition is satisfied, set that score to lam value
                nonconformity_scores  # otherwise keep the original score
                )
          
        # Check if there are any `inf` values in the scores for this image
        if torch.isinf(nonconformity_scores).any():
          images_with_inf.append(i)
        #if is_window: # if input to score functions are image windows rather than images, crop the scores accordingly to that window
        nonconformity_scores = nonconformity_scores.squeeze(0) # squeeze out extra dimension(s)
        #print(f"                   Shape of nonconformity_scores: {nonconformity_scores.shape}")
        nonconformity_scores = get_center_window(nonconformity_scores, w)
        all_nonconformity_scores.append(nonconformity_scores.flatten().cpu().numpy())
        
      # Flatten the scores back into a 1D vector for all images
      scores = np.concatenate(all_nonconformity_scores)#.reshape(-1,1)
      
    
      # Calculate the quartiles using NumPy
      q1 = np.percentile(scores, 25)
      median = np.percentile(scores, 50)
      q3 = np.percentile(scores, 75)
      max_value = np.max(scores)
    
      # Print the summary
      print(f"            Shape of scores: {scores.shape}")
      print(f"            Score distribution: Q1={q1}, Median={median}, Q3={q3}, Max={max_value}")
    
      # Print which images have `inf` values in their scores
      #if images_with_inf:
      #  print(f"            Indices of images with inf nonconformity scores: {images_with_inf}")
      #else:
      #  print("            No images have inf nonconformity scores.")
 
      return scores

      

    def nonconformity_score_inv_fn_modified(score_cutoffs, x):

        
        
        # Input: score_cutoffs, x; both should be 1D vectors of size (num_images * I * J)
        # Output: the (range of) label pixel intensities such that (pixel-wise), the condition
        # nonconformity_score_fn(x, Output) <= score_cutoffs

        # Check if x needs to be reshaped

     

      

      w = window_size
      x_full  = input_dataset[0][0].to(device)
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
      lb, fhat, ub = heuristic_psets[:,0,:,:,:],heuristic_psets[:,1,:,:,:],heuristic_psets[:,2,:,:,:]
      lhat = fhat + lb
      uhat = ub - fhat
      fhat = fhat.squeeze(0)
      lhat = lhat.squeeze(0)
      uhat = uhat.squeeze(0)
      #print(f"            Shape of lhat in inv score: {lhat.shape}")
      if isinstance(score_cutoffs, torch.Tensor):
        score_cutoffs = score_cutoffs.cpu().numpy()
      lhat = vectorize(get_center_window(lhat,w))[pixel_idx]#.numpy()
      fhat = vectorize(get_center_window(fhat,w))[pixel_idx]#.numpy()
      uhat = vectorize(get_center_window(uhat,w))[pixel_idx]#.numpy()


      # Calculate the lower and upper bounds for the labels for one image
      lower_bound = fhat - score_cutoffs * lhat
      upper_bound = fhat + score_cutoffs * uhat
      return [lower_bound,upper_bound]


      ############### DISCARDED due to condConf() not vectorized ##########
       # step 0: check input sizes
      # w = window_size
      # input_num_images = x.shape[0]/w**2
      # print(f"            Number of images in input to inverse score func: {input_num_images}")
      # input_num_scores_images = score_cutoffs.shape[0]/w**2
      # if (input_num_images != num_images):
      #   # Raise an exception and print an error message
      #   raise ValueError(f"Input size mismatch: Expected {num_images} images, but got {int(input_num_images)} images.")
      # if (input_num_scores_images != num_images):
      #   # Raise an exception and print an error message
      #   raise ValueError(f"Input size mismatch: Expected scores for {num_images} images, but got scores of {int(input_num_scores_images)} images.")
      # # if x.ndim == 1:
      # #     x = tensorize(x,num_images, I,J).to(device)
      # # else:
      # #     x = torch.tensor(x).to(device)
        
      # if score_cutoffs.ndim == 1:
      #     score_cutoffs = torch.tensor(score_cutoffs.reshape(num_images, w, w)).to(device)
      # else:
      #     score_cutoffs = torch.tensor(score_cutoffs).to(device)
      # for i in range(num_images):
      #   x_i = input_dataset[i][0].to(device)

      #   # Check the number of dimensions and adjust accordingly
      #   if x_i.ndim == 2:
      #     # If x_i is 2D (shape [320, 320]), add batch and channel dimensions
      #     x_i = x_i.unsqueeze(0).unsqueeze(0)  # Result: [1, 1, 320, 320]
      #   elif x_i.ndim == 3:
      #     # If x_i is 3D (shape [1, 320, 320] or [320, 320, 1]), add only the batch dimension
      #     if x_i.shape[0] != 1:  # If the first dimension is not the channel
      #       x_i = x_i.unsqueeze(0)  # Result: [1, 1, 320, 320] (assuming first dim is not the channel)
      #     else:
      #       x_i = x_i.unsqueeze(0)  # Result: [1, 1, 320, 320]
    
    
      #   score_cutoff_i = score_cutoffs[i]

      #       # Get the prediction set components for one image
      #   heuristic_psets = model(x_i)
      #   lb, fhat, ub = heuristic_psets[:,0,:,:,:],heuristic_psets[:,1,:,:,:],heuristic_psets[:,2,:,:,:]
      #   lhat = fhat + lb
      #   uhat = ub - fhat

      #   lhat = get_center_window(lhat,w)
      #   fhat = get_center_window(fhat,w)
      #   uhat = get_center_window(uhat,w)


      #       # Calculate the lower and upper bounds for the labels for one image
      #   lower_bound = fhat - score_cutoff_i * lhat
      #   upper_bound = fhat + score_cutoff_i * uhat

      #   all_lower_bounds.append(lower_bound.flatten().cpu().numpy())
      #   all_upper_bounds.append(upper_bound.flatten().cpu().numpy())

      # ub = np.concatenate(all_upper_bounds)
      # lb = np.concatenate(all_lower_bounds)
      # print(f"            Sizes of inverted nonconf scores: {[len(ub),len(lb)]}")
      # # Flatten the bounds back into 1D vectors for all images
      # return [ub,lb]
      ############### DISCARDED due to condConf() not vectorized ##########
    
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

        window_size = config["center_window_size"]
        variance_nbhd_size = config["var_window_size"]
        print(f"        Using the middle ({window_size}*{window_size}) windows of calib set")
        print(f"        Variance estimation window size: ({variance_nbhd_size}*{variance_nbhd_size})")
        print( "        Generate calib set for condConf.")
        x_calib = torch.stack([data[0] for data in dataset])
        print(f"            Length of x_calib:{len(x_calib)}")
        x_calib = generate_linear_phi_components(x_calib, window_size, variance_nbhd_size)
        y_calib = np.concatenate([vectorize(get_center_window(dataset[i][1],window_size)) for i in range(len(dataset))])


        condConf_basis_type = config['condConf_basis_type'] # possible values: "const", "linear"
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
        nonconformity_score_fn_modified, nonconformity_score_inv_fn_modified = create_nonconformity_score_fns_modified(model, dataset, window_size,  device,  lambdas)
        
        #print("Vectorize calib set for condConf")
        #downsampled_images = torch.stack([dataset[i][0] for i in range(len(dataset))])  # Stack all input downsampled images together
        #x_calib = vectorize(downsampled_images.cpu())
        #y_calib = vectorize(labels.cpu())
        #print(f"Length of x_calib (vectorized): {len(x_calib)}")
        
        print( "        Set up the condConf optimization problem")
        cond_conf = CondConf(score_fn  = nonconformity_score_fn_modified, Phi_fn = phi_fn, quantile_fn= None, infinite_params = infinite_params,seed=0)
        print(f"        Shapes of x_calib(with phi components) and y_calib: {[x_calib.shape, y_calib.shape]}")
        cond_conf.setup_problem(x_calib, y_calib)   



        print("    Applying split CP for calib loss table")
        # Retrieve the dataloader and get nonconformity scores
        nonconformity_scores = get_nonconformity_scores_from_dataset(model, dataset, device, lambdas,I, J, num_images)
        
        # Set the value of lam to be the 1 - (alpha / T) quantile of the nonconformity scores
        print("        Pick the adjusted conformal quantile")
        lam = torch.quantile(torch.tensor(nonconformity_scores), 1 - alpha / num_images)
        model.set_lhat(lam)  # set the lam value of model to the tuned lam value
        print("")
        print(f"            Model's lambdahat set to {model.lhat} by split CP")
        # Compute the loss table for each lambda value, 
        # starting from largest in grid and stopping at tunned lambda value

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







def get_nonconformity_scores_from_dataset(model, out_dataset, device, lambda_vec,I, J, num_images):
    # Create the nonconformity_score_fn with model, device, and lambda_vec
    nonconformity_score_fn, nonconformity_score_inv_fn = create_nonconformity_score_fns(model, device, lambda_vec,I, J, num_images)

    nonconformity_scores = []  # empty list for nonconformity scores to be appended to

    dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    model = model.to(device)  # load model to GPU
    #print("Printing nonconformity scores for each batch for debugging...")
    for batch in dataloader:
        x, labels = batch
        nonconformity_scores_batch = nonconformity_score_fn(x, labels)
        nonconformity_scores.append(torch.tensor(nonconformity_scores_batch, device=device))  # Append as tensor

    nonconformity_scores = torch.cat(nonconformity_scores, dim=0).detach().cpu().numpy()
    return nonconformity_scores


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
      #for i in tqdm(range(len(dataset))):
      #  labels[i] = dataset[i][1].cpu()
      #  outputs[i,:,:,:,:] = model(dataset[i][0].unsqueeze(0).to(device)).cpu()

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

        # Retrieve the dataloader and get nonconformity scores
        nonconformity_scores = get_nonconformity_scores_from_dataset(model, dataset, device, lambdas,I, J, num_images)
        
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












