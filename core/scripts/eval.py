import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from core.calibration.calibrate_model import get_rcps_loss_fn, get_rcps_metrics_from_outputs
from core.utils import standard_to_minmax

from core.calibration.calibrate_model_new import vectorize,  tensorize,  create_nonconformity_score_fns_modified,  generate_linear_phi_components, generate_linear_phi_components_NN, generate_const_phi_components, get_center_window
from core.condconf import CondConf
import wandb
import pdb

from joblib import Parallel, delayed

def transform_output(x,self_normalize=True):
  if self_normalize:
    x = x - x.min()
    x = x / x.max()

  x = np.maximum(0,np.minimum(255*x.cpu().squeeze(), 255))
  if len(x.shape) == 3:
    x = x.permute(1,2,0)
  return x.numpy().astype(np.uint8)


def process_pixel(model,
                  condConf,
                  config, 
                  lambdas, 
                  window_size, 
                  device, 
                  val_dataset_i, 
                  pixel_idx, 
                  x_vec):
    model = model.to(device)
    #print(f"condConf.predict pixel idx: {pixel_idx}")
    #print(f"x_vec shape: {x_vec.shape}",flush = True)
    handcrafted_basis = config["handcrafted_basis_components"]
    if config['condConf_basis_type'] != "linear":
      input_pixel = x_vec[pixel_idx]#.reshape(-1, 1)
    else:
      if handcrafted_basis != "handcrafted": # use NN basis
        num_basis_components = config["d"]
      else:
        num_basis_components = len(config['basis_components'])
      input_pixel = x_vec[pixel_idx,:].reshape(-1, num_basis_components)
    # Create inverse score function for the current pixel
    #print(f"            Create inv score function for {pixel_idx+1}-th pixel.")
    score_defn = config['score_defn']
    _, nonconformity_score_inv_fn_modified = create_nonconformity_score_fns_modified(model,
                                                                                    val_dataset_i, 
                                                                                    window_size, 
                                                                                    device, 
                                                                                    lambdas, 
                                                                                    pixel_idx,
                                                                                    score_defn)

    # Generate the CondConf prediction set
    condConf_pset = condConf.predict(
        quantile=config['alpha'], 
        x_test=input_pixel, 
        score_inv_fn=nonconformity_score_inv_fn_modified, 
        exact=False,  # False
        randomize=False # False
    )

    # save CUDA memory
    del model
    torch.cuda.empty_cache() 
    return condConf_pset[0], condConf_pset[1]


def get_images_condConf_parallel(model, 
                                val_dataset, 
                                device, 
                                idx_iterator, 
                                config, 
                                lambdas, 
                                condConf):
    # Parallelized version of get_images_condConf() to speed up
    print(f"Entered condConf pset generation for validation set.")
    with torch.no_grad():
        model = model.to(device)
        window_size = config["val_center_window_size"]
        if model.lhat == None:
          if config["uncertainty_type"] != "softmax":
            lam = 1.0
          else:
            lam = 0.99

        try:
            # If dataset is iterable, create a list of outputs
            my_iter = iter(val_dataset)
            print(f"Number of val images is: {len(idx_iterator)}")
            val_dataset = [next(my_iter) for img_idx in idx_iterator]
        except:
            pass

        num_val_images = len(val_dataset)
        print("    Generate val set for condConf.")
        x_test = torch.stack([data[0] for data in val_dataset])

        condConf_basis_type = config['condConf_basis_type']  # possible values: "const", "linear"
        if condConf_basis_type != "linear":
            x_test = generate_const_phi_components(x_test, window_size)
        else:
            handcrafted_basis = config["handcrafted_basis_components"]
            if handcrafted_basis != "handcrafted": # use NN basis
              d = config["d"]
              x_test = x_test.to(device)
              x_test = generate_linear_phi_components_NN(x_test, device, window_size, d)
            else:  # use handcrafted basis
              basis_components = config["basis_components"]
              variance_nbhd_size = config["var_window_size"]
               # Initialize optional lists only when required
              all_variances = [] if ("var" in config["basis_components"]) and (condConf_basis_type == "linear") else None
              all_blurred_vars = [] if ("blurred_var" in config["basis_components"]) and (condConf_basis_type == "linear") else None
              all_blurred_xijs = [] if ("blurred_xij" in config["basis_components"]) and (condConf_basis_type == "linear") else None
              x_test = generate_linear_phi_components(x_test, window_size,variance_nbhd_size,basis_components)
        
        y_test = np.concatenate([vectorize(get_center_window(data[1], window_size)) for data in val_dataset])

        condConf_psets = []
        examples_output = []

       
        print("    Computing condConf psets for val set.")
        length = window_size ** 2  # number of pixels in the current image

        # Loop through validation dataset
        for img_idx in idx_iterator:
            print(f"    Compute pset for the {img_idx+1}-th val image")
            start_idx = img_idx * length 
            end_idx = start_idx + length
            x_vec = x_test[start_idx:end_idx, :]  # get the phi components for the (img_idx)-th input image


            if (handcrafted_basis == "handcrafted") and (condConf_basis_type == "linear"):

              # Compute variances if required
              if all_variances is not None:
                  variance_component_idx = config["basis_components"].index("var")
                  variances = x_vec[:, variance_component_idx]
                  variances = tensorize(variances, 1, window_size, window_size)
                  all_variances.append(variances.cpu().numpy())

              # Compute blurred_var if required
              if all_blurred_vars is not None:
                  blurred_var_component_idx = config["basis_components"].index("blurred_var")
                  blurred_vars = x_vec[:, blurred_var_component_idx]
                  blurred_vars = tensorize(blurred_vars, 1, window_size, window_size)
                  all_blurred_vars.append(blurred_vars.cpu().numpy())

              # Compute blurred_xij if required
              if all_blurred_xijs is not None:
                  blurred_xij_component_idx = config["basis_components"].index("blurred_xij")
                  blurred_xijs = x_vec[:, blurred_xij_component_idx]
                  blurred_xijs = tensorize(blurred_xijs, 1, window_size, window_size)
                  all_blurred_xijs.append(blurred_xijs.cpu().numpy())

            # Get (img_idx)-th validation image
            val_dataset_i = val_dataset[img_idx]

            # Parallelize the pixel processing
            results = Parallel(n_jobs=4)(delayed(process_pixel)(
                model, condConf, config, lambdas, window_size, device, val_dataset_i, pixel_idx, x_vec
            ) for pixel_idx in range(length))

            all_lower_bounds, all_upper_bounds = zip(*results)  # "*" unpacks the list results, and zip() combines elements from each tuple at corresponding positions

            # Tensorize the condConf prediction set to image forms
            condConf_pset_lower = np.concatenate(all_lower_bounds)
            condConf_pset_upper = np.concatenate(all_upper_bounds)
            condConf_lower_bound = tensorize(condConf_pset_lower, 1, window_size, window_size)
            condConf_upper_bound = tensorize(condConf_pset_upper, 1, window_size, window_size)

            # Get model's prediction on the downsampled input
            x_full = val_dataset[img_idx][0].unsqueeze(0).to(device)
            if config["model"] == "trivialModel":
                model_prediction = x_full
            else:
                model_prediction = model(x_full)[:, 1, :, :, :]  # Get model prediction

            # Apply the get_center_window function to the model prediction
            example_output = get_center_window(model_prediction.squeeze(), window_size)

            # Combine the lower, middle, and upper predictions into a single tensor
            combined_output = torch.stack([condConf_lower_bound.cpu(), example_output.unsqueeze(0).cpu(), condConf_upper_bound.cpu()], dim=0)
            examples_output.append(combined_output)

        examples_output = [torch.stack([example[0], example[1], example[2]], dim=0) for example in examples_output]
        examples_gt = [val_dataset[img_idx][1] for img_idx in idx_iterator]
        inputs = [val_dataset[img_idx][0] for img_idx in idx_iterator]

        # Build the raw_images_dict
        raw_images_dict = {
            'inputs': inputs,
            'gt': examples_gt,
            'predictions': [example[1] for example in examples_output],
            'condConf_lower_edge': [example[0] for example in examples_output],
            'condConf_upper_edge': [example[2] for example in examples_output],
        }

        # Add optional components only if they were created
        if (handcrafted_basis == "handcrafted") and (condConf_basis_type == "linear"):
          if all_variances is not None:
              raw_images_dict['variances'] = [variance for variance in all_variances]
          if all_blurred_vars is not None:
              raw_images_dict['blurred_var'] = [blurred_var for blurred_var in all_blurred_vars] 
          if all_blurred_xijs is not None:
              raw_images_dict['blurred_xij'] = [blurred_xij for blurred_xij in all_blurred_xijs]  

        try:
            val_dataset.reset()
        except AttributeError:
            print("Warning: val_dataset does not have a reset method.")

        return raw_images_dict





def get_images_condConf_non_parallel(model, 
                                     val_dataset, 
                                     device, 
                                     idx_iterator, 
                                     config, 
                                     lambdas, 
                                     condConf):
    print(f"Entered condConf pset generation for validation set.")
    with torch.no_grad():
        model = model.to(device)
        window_size = config["val_center_window_size"]
        if model.lhat == None:
            if config["uncertainty_type"] != "softmax":
                lam = 1.0
            else:
                lam = 0.99

        try:
            my_iter = iter(val_dataset)
            print(f"Number of val images is: {len(idx_iterator)}")
            val_dataset = [next(my_iter) for img_idx in idx_iterator]
        except:
            pass

        num_val_images = len(val_dataset)
        print("    Generate val set for condConf.")
        x_test = torch.stack([data[0] for data in val_dataset])

        condConf_basis_type = config['condConf_basis_type']
        if condConf_basis_type != "linear":
            x_test = generate_const_phi_components(x_test, window_size)
        else:
            handcrafted_basis = config["handcrafted_basis_components"]
            if handcrafted_basis != "handcrafted":
                d = config["d"]
                x_test = x_test.to(device)
                x_test = generate_linear_phi_components_NN(x_test, device, window_size, d)
            else:
                basis_components = config["basis_components"]
                variance_nbhd_size = config["var_window_size"]
                x_test = generate_linear_phi_components(x_test, window_size, variance_nbhd_size, basis_components)

        y_test = np.concatenate([vectorize(get_center_window(data[1], window_size)) for data in val_dataset])

        condConf_psets = []
        examples_output = []

        print("    Computing condConf psets for val set.")
        length = window_size ** 2

        for img_idx in idx_iterator:
            print(f"    Compute pset for the {img_idx+1}-th val image")
            start_idx = img_idx * length
            end_idx = start_idx + length
            x_vec = x_test[start_idx:end_idx, :]

            if (handcrafted_basis == "handcrafted") and (condConf_basis_type == "linear"):
                if "var" in config["basis_components"]:
                    variance_component_idx = config["basis_components"].index("var")
                    variances = x_vec[:, variance_component_idx]
                    variances = tensorize(variances, 1, window_size, window_size)
                if "blurred_var" in config["basis_components"]:
                    blurred_var_component_idx = config["basis_components"].index("blurred_var")
                    blurred_vars = x_vec[:, blurred_var_component_idx]
                    blurred_vars = tensorize(blurred_vars, 1, window_size, window_size)
                if "blurred_xij" in config["basis_components"]:
                    blurred_xij_component_idx = config["basis_components"].index("blurred_xij")
                    blurred_xijs = x_vec[:, blurred_xij_component_idx]
                    blurred_xijs = tensorize(blurred_xijs, 1, window_size, window_size)

            val_dataset_i = val_dataset[img_idx]

            all_lower_bounds = []
            all_upper_bounds = []

            for pixel_idx in range(length):
                lower_bound, upper_bound = process_pixel(
                    model, condConf, config, lambdas, window_size, device, val_dataset_i, pixel_idx, x_vec
                )
                all_lower_bounds.append(lower_bound)
                all_upper_bounds.append(upper_bound)

            condConf_pset_lower = np.concatenate(all_lower_bounds)
            condConf_pset_upper = np.concatenate(all_upper_bounds)
            condConf_lower_bound = tensorize(condConf_pset_lower, 1, window_size, window_size)
            condConf_upper_bound = tensorize(condConf_pset_upper, 1, window_size, window_size)

            x_full = val_dataset[img_idx][0].unsqueeze(0).to(device)
            if config["model"] == "trivialModel":
                model_prediction = x_full
            else:
                model_prediction = model(x_full)[:, 1, :, :, :]

            example_output = get_center_window(model_prediction.squeeze(), window_size)
            combined_output = torch.stack([
                condConf_lower_bound.cpu(), example_output.unsqueeze(0).cpu(), condConf_upper_bound.cpu()
            ], dim=0)
            examples_output.append(combined_output)

        examples_output = [torch.stack([example[0], example[1], example[2]], dim=0) for example in examples_output]
        examples_gt = [val_dataset[img_idx][1] for img_idx in idx_iterator]
        inputs = [val_dataset[img_idx][0] for img_idx in idx_iterator]

        raw_images_dict = {
            'inputs': inputs,
            'gt': examples_gt,
            'predictions': [example[1] for example in examples_output],
            'condConf_lower_edge': [example[0] for example in examples_output],
            'condConf_upper_edge': [example[2] for example in examples_output],
        }

        if (handcrafted_basis == "handcrafted") and (condConf_basis_type == "linear"):
            if "var" in config["basis_components"]:
                raw_images_dict['variances'] = [variance for variance in all_variances]
            if "blurred_var" in config["basis_components"]:
                raw_images_dict['blurred_var'] = [blurred_var for blurred_var in all_blurred_vars]
            if "blurred_xij" in config["basis_components"]:
                raw_images_dict['blurred_xij'] = [blurred_xij for blurred_xij in all_blurred_xijs]

        try:
            val_dataset.reset()
        except AttributeError:
            print("Warning: val_dataset does not have a reset method.")

        return raw_images_dict





def get_images(model,
               val_dataset,
               device,
               idx_iterator,
               config):
  with torch.no_grad():
    model = model.to(device)
    window_size = config["val_center_window_size"]
    lam = None
    if model.lhat == None:
      if config["uncertainty_type"] != "softmax":
        lam = 1.0
      else:
        lam = 0.99

    try:
      # If dataset is iterable, create a list of outputs
      my_iter = iter(val_dataset)
      val_dataset = [next(my_iter) for img_idx in idx_iterator]
    except:
      pass


    examples_output = []
    for img_idx in idx_iterator:
      CP_lb, model_pred, CP_ub = model.nested_sets((val_dataset[img_idx][0].unsqueeze(0).to(device),),lam=lam) # lam = 1 or 0.99 because nested_sets() will use lhat which is tuned by CP, and 1*lhat = lhat.
      
      CP_lb = get_center_window(CP_lb.squeeze(),window_size)
      model_pred = get_center_window(model_pred.squeeze(),window_size)
      CP_ub = get_center_window(CP_ub.squeeze(),window_size)
      combined_output = torch.stack([CP_lb.cpu(), model_pred.cpu(), CP_ub.cpu()], dim=0)

      # Append the tensor to the list
      #condConf_psets.append(combined_output)
      #examples_output.append(example_output)

      examples_output.append(combined_output)

    examples_output = [torch.stack([example[0], example[1], example[2]], dim=0) for example in examples_output]
    #examples_output = [model.nested_sets((val_dataset[img_idx][0].unsqueeze(0).to(device),),lam=lam) for img_idx in idx_iterator]
    examples_gt = [val_dataset[img_idx][1] for img_idx in idx_iterator]
    if val_dataset[0][0].shape[0] > 1:
      inputs = [val_dataset[img_idx][0][0] for img_idx in idx_iterator]
    else:
      inputs = [val_dataset[img_idx][0] for img_idx in idx_iterator]
    raw_images_dict = {'CP_lower_edge': [example[0] for example in examples_output], 
                       'CP_upper_edge': [example[2] for example in examples_output] 
                      }

    if val_dataset[0][0].shape[0] > 1:
      examples_input = [wandb.Image(transform_output(val_dataset[img_idx][0][0])) for img_idx in idx_iterator]
    else:
      examples_input = [wandb.Image(transform_output(val_dataset[img_idx][0])) for img_idx in idx_iterator]
    examples_lower_edge = [wandb.Image(transform_output(example[0])) for example in examples_output]
    examples_prediction = [wandb.Image(transform_output(example[1])) for example in examples_output]
    examples_upper_edge = [wandb.Image(transform_output(example[2])) for example in examples_output]
    examples_ground_truth = [wandb.Image(transform_output(val_dataset[img_idx][1])) for img_idx in idx_iterator]

    # Calculate lengths on their own scales
    lower_lengths = [example[1]-example[0] for example in examples_output]
    lower_lengths = [transform_output(lower_lengths[i]/(examples_output[i][1].max()-examples_output[i][1].min()), self_normalize=False) for i in range(len(examples_output))]
    #lower_lengths = [lower_lengths[i]/(examples_output[i][1].max()-examples_output[i][1].min()) for i in range(len(examples_output))]
    upper_lengths = [example[2]-example[1] for example in examples_output]
    upper_lengths = [transform_output(upper_lengths[i]/(examples_output[i][1].max()-examples_output[i][1].min()), self_normalize=False) for i in range(len(examples_output))]
    #upper_lengths = [upper_lengths[i]/(examples_output[i][1].max()-examples_output[i][1].min()) for i in range(len(examples_output))]

    examples_lower_length = [wandb.Image(ll) for ll in lower_lengths]
    examples_upper_length = [wandb.Image(ul) for ul in upper_lengths]

    try:
      val_dataset.reset()
    except:
      pass

    return raw_images_dict
    #examples_input, examples_lower_edge, examples_prediction, examples_upper_edge, examples_ground_truth, examples_lower_length, examples_upper_length, raw_images_dict

def get_loss_table(model, dataset, config):
  try:
    dataset.reset()
  except:
    print("dataset is map-style (not resettable)")
  with torch.no_grad():
    if config["uncertainty_type"] == "softmax":
      lambdas = torch.linspace(config['minimum_lambda_softmax'],config['maximum_lambda_softmax'],config['num_lambdas'])
    else:
      lambdas = torch.linspace(config['minimum_lambda'],config['maximum_lambda'],config['num_lambdas'])
    model.eval()
    device = config['device']
    rcps_loss_fn = get_rcps_loss_fn(config)
    model = model.to(device)
    labels = torch.cat([x[1].unsqueeze(0).to(device).to('cpu') for x in dataset], dim=0).cpu()

    if config['dataset'] == 'temca':
      outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).to('cpu') for x in dataset], dim=0)
    else:
      outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
      outputs_shape[0] = len(dataset)
      outputs = torch.zeros(tuple(outputs_shape),device='cpu')
      
      for i in range(len(dataset)):
        print(f"Validation output {i}")
        outputs[i,:,:,:,:] = model(dataset[i][0].unsqueeze(0).to(device)).cpu()
    out_dataset = TensorDataset(outputs,labels)

    print("GET LOSS TABLE FROM OUTPUTS")
    batch_size = 4
    num_samples = len(out_dataset)
    num_lambdas = config['num_lambdas']
    
    # Determine the number of full batches
    num_full_batches = num_samples // batch_size
    adjusted_num_samples = num_full_batches * batch_size
    
    # Initialize the loss_table with the appropriate shape
    loss_table = torch.zeros((adjusted_num_samples, num_lambdas), device=device)
    
    dataloader = DataLoader(out_dataset, batch_size=4, shuffle=False, num_workers=0) 
    model = model.to(device)
    i = 0
    for batch in dataloader:
      x, labels = batch
        
      # Skip the last batch if it is smaller than batch_size
      if x.shape[0] < batch_size: 
          break
        
      x = x.to(device)
      labels = labels.to(device)
        
      # For each lambda value, compute the sets and loss
      for j in range(lambdas.shape[0]):
          sets = model.nested_sets_from_output(x, lam=lambdas[j]) 
          #print(x.shape[0])
            
          # Calculate the loss
          loss = rcps_loss_fn(sets, labels)
          #print(loss.shape)
          #print('\n')
            
          # Ensure the loss is correctly assigned to loss_table
          loss_table[i:i + x.shape[0], j] = loss
        
      i += x.shape[0]
    
  print("DONE!")
  return loss_table


def eval_set_metrics(model, dataset, config):
  try:
    dataset.reset()
  except:
    print("dataset is map-style (not resettable)")
  with torch.no_grad():
    model.eval()
    device = config['device']
    rcps_loss_fn = get_rcps_loss_fn(config)
    model = model.to(device)
    labels = torch.cat([x[1].unsqueeze(0).to(device).to('cpu') for x in dataset], dim=0).cpu()

    if config['dataset'] == 'temca':
      outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).to('cpu') for x in dataset], dim=0)
    else:
      outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
      outputs_shape[0] = len(dataset)
      outputs = torch.zeros(tuple(outputs_shape),device='cpu')
      
      for i in range(len(dataset)):
        print(f"Validation output {i}")
        outputs[i,:,:,:,:] = model(dataset[i][0].unsqueeze(0).to(device)).cpu()
    out_dataset = TensorDataset(outputs,labels)

    print("GET RCPS METRICS FROM OUTPUTS")
    losses, sizes, spearman, stratified_risks, mse, spatial_miscoverage = get_rcps_metrics_from_outputs(model, out_dataset, rcps_loss_fn, device)
    print("DONE!")
    return losses.mean(), sizes, spearman, stratified_risks, mse, spatial_miscoverage





def eval_net(net, loader, device):
    with torch.no_grad():
      net.eval()
      net.to(device=device)
      #label_type = torch.float32 if net.n_classes == 1 else torch.long

      val_loss = 0
      num_val = 0

      with tqdm(total=10000, desc='Validation round', unit='batch', leave=False) as pbar:
          for batch in loader:
              labels = batch[-1].to(device=device)
              x = tuple([batch[i] for i in range(len(batch)-1)])
              x = [x[i].to(device=device, dtype=torch.float32) for i in range(len(x))]

              # Predict
              labels_pred = net(*x) # Unpack tuple

              num_val += labels.shape[0]
              val_loss += net.loss_fn(labels_pred, labels).item()
              pbar.update()

      net.train()

      if num_val == 0:
        return 0

      return val_loss/num_val
