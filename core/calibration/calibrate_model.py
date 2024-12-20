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







def get_nonconformity_scores_from_outputs(model, out_dataset, device, lambda_vec):
    # Convert lambda_vec to a PyTorch tensor and move it to the correct device
    if isinstance(lambda_vec, np.ndarray):
        lambdas = torch.tensor(lambda_vec, dtype=torch.float32).to(device)
    else:
        lambdas = lambda_vec.to(device)

    nonconformity_scores = []  # Empty list for nonconformity scores to be appended to

    dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    model = model.to(device)  # Load model to GPU
    print("Printing nonconformity scores for each batch for debugging...")
    for batch in dataloader:
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        #print(f"Shape of labels: {labels.shape}")

        # Step 1: Get heuristic prediction set components: f, l, u
        heuristic_psets = model.nested_sets(x, 1)  # lambda = 1 since we just want the heuristic prediction set
        lb, fhat, ub = heuristic_psets  # lower-hat, f-hat, upper-hat
        lhat = fhat + lb
        uhat = ub - fhat
        #print(f"Shape of lhat: {lhat.shape}")


        # Initialize a tensor to store nonconformity scores for the current batch,
        # starting from inf since nonconformity scores are defined by min...
        batch_nonconformity_scores = torch.full(labels.shape, float('inf'), dtype=torch.float, device=device)

        # Step 2: Compute pixel-wise nonconformity scores
        for lam in lambdas:  # Traverse lambda values from smallest to largest
            # Check the property for the current lambda value
            property_satisfied = containment_check(lam, lhat, fhat, uhat, labels)
            # Ensure the condition tensor is on the same device as batch_nonconformity_scores
            property_satisfied = property_satisfied.to(device)
            # Update nonconformity scores where the property is satisfied for the first time
            batch_nonconformity_scores = torch.where(
                (batch_nonconformity_scores == float('inf')) & property_satisfied,  # only need to look at scores that are still inf
                lam,  # if containment condition is satisfied, set that score to lam value
                batch_nonconformity_scores  # otherwise keep the original score
            )
        # After the loop, check if the nonconformity scores were updated
        print(f"Min score after update: {batch_nonconformity_scores.min().item()}")
        print(f"Max score after update: {batch_nonconformity_scores.max().item()}")
        print(f"Mean score after update: {batch_nonconformity_scores.mean().item()}")
        nonconformity_scores = nonconformity_scores + [batch_nonconformity_scores]

    nonconformity_scores = torch.cat(nonconformity_scores, dim=0).detach().cpu().numpy() #maybe no need to convert to tensor
    return nonconformity_scores

def containment_check(lam, lhat, fhat, uhat, labels):
    # Check if label is within the interval [fhat - lambda * lhat, fhat + lambda * uhat]
    lower_bound = fhat - lam * lhat
    upper_bound = fhat + lam * uhat
    containment = (labels >= lower_bound) & (labels <= upper_bound)
    return containment





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
    loss = rcps_loss_fn(sets, labels)
    print(f"Size of loss:{loss.shape}")
    losses = losses + [loss,]
    sets_full = (sets[2]-sets[0]).flatten(start_dim=1).detach().cpu().numpy()
    #sets_full = (sets[2]-sets[0]).squeeze().detach().cpu().numpy()
    print(f"Size of sets_full:{sets_full.shape}")
    size_random_idxs = np.random.choice(sets_full.shape[1],size=loss.shape[0])
    #size_random_idxs = np.random.choice(sets_full.shape[1],size=sets_full.shape[0])
    
    size_samples = sets_full[range(sets_full.shape[0]),size_random_idxs]
    #residuals = residuals + [(labels - sets[1]).abs().flatten(start_dim=1)[range(loss.shape[0]),size_random_idxs]]
    
    residuals = residuals + [(labels - sets[1]).abs().flatten(start_dim=1)[range(sets_full.shape[0]),size_random_idxs]]
    
    spatial_miscoverages = spatial_miscoverages + [(labels > sets[2]).float() + (labels < sets[0]).float()]
    sizes = sizes + [torch.tensor(size_samples),]
  losses = torch.cat(losses,dim=0)
  #losses = losses.T
  sizes = torch.cat(sizes,dim=0)
  sizes = sizes + torch.rand(size=sizes.shape).to(sizes.device)*1e-6
  print(f"Size of sizes:{sizes.shape}")
  residuals = torch.cat(residuals,dim=0).detach().cpu().numpy() 
  spearman = spearmanr(residuals, sizes)[0]
  mse = (residuals*residuals).mean().item()
  spatial_miscoverage = torch.cat(spatial_miscoverages, dim=0).detach().cpu().numpy().mean(axis=0).mean(axis=0)
  size_bins = torch.tensor([0, torch.quantile(sizes, 0.25), torch.quantile(sizes, 0.5), torch.quantile(sizes, 0.75)])
  buckets = torch.bucketize(sizes, size_bins)-1
  print(f"Size of losses:{losses.shape}")
  print(f"Size of buckets:{buckets.shape}")
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
  print(f"Shape of misses in fraction_missed_loss(): {misses.shape}")
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
            T_value = labels_shape[0]
            print(f"Number of data points (T) in calibration set: {T_value}")
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
        nonconformity_scores = get_nonconformity_scores_from_outputs(model, out_dataset, device, lambdas)
        
        # Set the value of lam to be the 1 - (alpha / T) quantile of the nonconformity scores
        # Print the shape of the nonconformity scores
        print(f"Shape of one batch of nonconformity scores: {np.array(nonconformity_scores).shape}")
        print("Pick the adjusted conformal quantile")
        lam = np.quantile(nonconformity_scores, 1 - alpha / T_value)
        lam = torch.tensor(lam)
        model.set_lhat(lam)  # set the lam value of model to the tuned lam value
        print("")
        print(f"Model's lambdahat is set to {model.lhat}")

        # Compute the loss table for each lambda value
        print("Computing loss table")
        calib_loss_table = torch.zeros((outputs.shape[0], lambdas.shape[0]), device='cpu')
        rcps_loss_fn = get_rcps_loss_fn(config)
        for lamd_value in reversed(lambdas):
          losses = get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lamd_value, device)
              # losses = a vector of losses (=missed fractions) on each calibration image when inflator = lam-dlambda
          calib_loss_table[:,np.where(lambdas==lamd_value)[0]] = losses[:,None]
              # fill in loss table (to keep track of the lambdahat tunning procedure)
          if lamd_value == lam:
            break
    print("DONE!")
    return model, calib_loss_table





