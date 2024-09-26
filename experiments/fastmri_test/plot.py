import os, sys, io, pathlib
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pickle as pkl
from core.calibration.calibrate_model import evaluate_from_loss_table
from core.scripts.eval import transform_output 
from tqdm import tqdm
from PIL import Image,ImageDraw, ImageFont
import pdb
import shutil
import cmasher as cmr


# for condConf modifications
from core.calibration.calibrate_model_new import vectorize, tensorize, create_nonconformity_score_fns_modified, generate_linear_phi_components,get_center_window
from core.scripts.router import get_config, get_img_generating_fname,get_img_save_fname


def normalize_01(x):
  x = x - x.min()
  x = x / x.max()
  return x

class CPU_Unpickler(pkl.Unpickler):
  def find_class(self, module, name):
    if module == 'torch.storage' and name == '_load_from_bytes':
      return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
    else:
      return super().find_class(module, name)

def plot_mse(methodnames,results_list):
  def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
formatted as -.4)."""
    val_str = '{:g}'.format(x)
    val_str = val_str if len(val_str) <= 5 else val_str[:5]
    if np.abs(x) > 0 and np.abs(x) < 1:
      return val_str.replace("0", "", 1)
    else:
      return val_str
  major_formatter = ticker.FuncFormatter(my_formatter)

  plt.figure(figsize=(12,1.75))
  sns.set(font_scale=1.2) # 1col scaling
  sns.set_style("white")
  sns.set_palette('pastel')
  # Crop sizes to 99%
  mses = np.array([results['mse'] for results in results_list])
  #methodnames = ['Residual Magnitude (RM)', 'Gaussian (G)', 'Softmax (S)', 'Quantile Regression (QR)'] # For ICML only!
  #df = pd.DataFrame({'Spearman Rank Correlation' : [results['spearman'] for results in results_list], 'Method': [method.replace(' ','\n') for method in methodnames]})
  #g = sns.scatterplot(data=df, x='Method', y='Spearman Rank Correlation', kind='bar')
  for j in range(len(methodnames)):
    plt.scatter(x=[mses[j],], y=[np.random.uniform(size=(1,))/4,], s=70, label=methodnames[j])
  sns.despine(top=True, bottom=True, right=True, left=True)
  plt.gca().set_yticks([])
  plt.gca().set_yticklabels([])
  plt.ylim([-0.1,1])
  plt.xlim([0,None])
  plt.legend(bbox_to_anchor=(-0.5, 0.5))
  plt.gca().tick_params(axis=u'both', which=u'both',length=0)
  plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
  plt.locator_params(axis="x", nbins=4)
  plt.xlabel("Mean-squared error of prediction")
  plt.tight_layout()
  plt.savefig('outputs/fastmri-mse.pdf',bbox_inches="tight")

def plot_spearman(methodnames,results_list):
  plt.figure(figsize=(12,1.75))
  sns.set_palette('pastel')
  # Crop sizes to 99%
  spearmans = [results['spearman'] for results in results_list]
  #df = pd.DataFrame({'Spearman Rank Correlation' : [results['spearman'] for results in results_list], 'Method': [method.replace(' ','\n') for method in methodnames]})
  #g = sns.scatterplot(data=df, x='Method', y='Spearman Rank Correlation', kind='bar')
  for j in range(len(methodnames)):
    plt.scatter(x=spearmans[j], y=[0,], s=70, label=methodnames[j])
  sns.despine(top=True, bottom=True, right=True, left=True)
  plt.gca().set_yticks([])
  plt.gca().set_yticklabels([])
  plt.ylim([-0.1,1])
  plt.legend(bbox_to_anchor=(-0.5, 0.5))
  plt.gca().tick_params(axis=u'both', which=u'both',length=0)
  plt.xlabel("Spearman rank correlation between heuristic and true residual")
  plt.tight_layout()
  plt.savefig('outputs/fastmri-spearman.pdf',bbox_inches="tight")

def plot_size_violins(methodnames,results_list):
  plt.figure(figsize=(5,5))
  sns.set(font_scale=1.2) # 1col format
  #sns.set(font_scale=2) # 2col format
  sns.set_style("white")
  sns.set_palette('pastel')
  #methodnames = ['RM', 'G', 'S', 'QR'] # For ICML only!
  # Crop sizes to 99%
  for results in results_list:
    results['sizes'] = torch.clamp(results['sizes'], min=0, max=2) + (torch.rand(results['sizes'].shape)-0.5)*0.01

  df = pd.DataFrame({'Interval Length' : torch.cat([results['sizes'] for results in results_list]).tolist(), 'Method': [method.replace(' ','\n') for method in methodnames for i in range(results_list[0]['sizes'].shape[0])]})
  g = sns.violinplot(data=df, x='Method', y='Interval Length', cut=0)
  sns.despine(top=True, right=True)
  plt.yticks([0,.5/5.0,1.0/5.0])
  plt.ylim([0,1.0/5.0])
  plt.xlabel('')
  plt.gca().set_yticklabels(['0%','5%','10%'])
  plt.tight_layout()
  plt.savefig('outputs/fastmri-sizes.pdf',bbox_inches="tight")

def plot_ssr(methodnames,results_list,alpha):
  plt.figure(figsize=(4,4))
  sns.set(font_scale=1.2) # 1col format
  #sns.set(font_scale=2) # 2col format
  sns.set_style("white")
  sns.set_palette(sns.light_palette("salmon"))
  #methodnames = ['RM', 'G', 'S', 'QR'] # For ICML only!
  df = pd.DataFrame({'Interval Length': len(results_list)*['Short', 'Short-Medium', 'Medium-Long', 'Long'], 'Size-Stratified Risk' : torch.cat([results['size-stratified risk'] for results in results_list]).tolist(), 'Method': [method.replace(' ','\n') for method in methodnames for i in range(results_list[0]['size-stratified risk'].shape[0])]})
  g = sns.catplot(data=df, kind='bar', x='Method', y='Size-Stratified Risk', hue='Interval Length',legend=False)
  sns.despine(top=True, right=True)
  plt.legend(loc='upper right') # 1col format
  #plt.legend(loc='upper right', fontsize=18) # 2col format
  plt.xlabel('')
  plt.ylim([None,0.25])
  plt.locator_params(axis="y", nbins=5)
  #plt.gca().axhline(y=alpha, color='#888888', linewidth=2, linestyle='dashed')
  #plt.text(2,alpha+0.005,r'$\alpha$',color='#888888')
  plt.tight_layout()
  plt.savefig('outputs/fastmri-size-stratified-risk.pdf',bbox_inches="tight")

def plot_risks(methodnames,loss_table_list,n,alpha,delta,num_trials=100): 
  fname = 'outputs/raw/risks.pth'
  if os.path.exists(fname):
    with open(fname, 'rb') as f:
      risks_list = pkl.load(f)
  else: 
    risks_list = []
    for loss_table in loss_table_list:
      risks = torch.zeros((num_trials,))
      for trial in tqdm(range(num_trials)):
        risks[trial] = evaluate_from_loss_table(loss_table,n,alpha,delta)
      risks_list = risks_list + [risks,]
    with open(fname, 'wb') as f:
      pkl.dump(risks_list,f)
  plt.figure(figsize=(5,5))
  sns.set(font_scale=1.2) # 1col format
  #sns.set(font_scale=2) # 2col format
  sns.set_style("white")
  sns.set_palette('pastel')
  #methodnames = ['RM', 'G', 'S', 'QR'] # For ICML only!
  df = pd.DataFrame({'Method' : [method.replace(' ','\n') for method in methodnames for i in range(num_trials)], 'Risk' : torch.cat(risks_list,dim=0).tolist()})
  g = sns.violinplot(data=df, x='Method', y='Risk')
  plt.gca().axhline(y=alpha, color='#888888', linewidth=2, linestyle='dashed')
  sns.despine(top=True, right=True)
  plt.ylim([0.07,None])
  plt.xlabel('')
  plt.locator_params(axis="y", nbins=5)
  plt.text(2.2,alpha-0.0018,r'$\alpha$',color='#888888')
  plt.tight_layout()
  plt.savefig('outputs/fastmri-risks.pdf',bbox_inches="tight")




# def get_img_save_fname(config): # the name of the folder 
#   results_fname = (f"/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_test/outputs/images/{config['dataset']}_"
#                  f"({config['num_calib_img']})_"
#                  f"({config['num_val_img']})_"
#                  f"{config['center_window_size']}_"
#                  f"{config['val_center_window_size']}_"
#                  f"{config['condConf_basis_type']}") # 
#   return results_fname

def plot_correlation_from_results(results):
    # Define the output directory based on config
    config = get_config()
    output_base_dir = get_img_save_fname(config)
    output_dir = os.path.join(output_base_dir, "sizePredCorr/")  # Treat sizePredCorr as a subfolder

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    condconf_basis_type = config['condConf_basis_type']

    # Unwrap the lists into flattened arrays
    predictions = np.hstack([pred.flatten() for pred in results['predictions']])
    condConf_lower = np.hstack([edge.flatten() for edge in results['condConf_lower_edge']])
    condConf_upper = np.hstack([edge.flatten() for edge in results['condConf_upper_edge']])
    
    # Compute condConf sizes (upper - lower bounds)
    condConf_sizes = condConf_upper - condConf_lower
    
    # Compute relative absolute error for each image individually
    all_rel_abs_errors = []
    for i in range(len(results['gt'])):
        gt = results['gt'][i]
        inputs = results['inputs'][i]
        
        # Apply the same window cropping to gt and inputs to match predictions
        gt_cropped = get_center_window(gt, results['predictions'][i].shape[-1])  # Ensure same shape as predictions
        inputs_cropped = get_center_window(inputs, results['predictions'][i].shape[-1])
        
        abs_error = np.abs(gt_cropped - inputs_cropped).flatten()
        max_val = np.maximum(np.abs(gt_cropped), np.abs(inputs_cropped)).flatten()
        rel_abs_error = abs_error / max_val
        all_rel_abs_errors.extend(rel_abs_error)  # Concatenate rel_abs_error for each image

    # Convert all_rel_abs_errors to a numpy array
    all_rel_abs_errors = np.array(all_rel_abs_errors)
    
    # Debugging: Print lengths of the arrays to ensure they match
    print(f"Length of condConf_sizes: {len(condConf_sizes)}")
    print(f"Length of predictions: {len(predictions)}")
    print(f"Length of all_rel_abs_errors: {len(all_rel_abs_errors)}")

    # Generate caption for plots
    config_caption = generate_config_caption()

    # 1. Scatter plot: condConf_sizes vs predictions
    if len(condConf_sizes) == len(predictions):
        df_1 = pd.DataFrame({
            'condConf_sizes': condConf_sizes,
            'predictions': predictions
        })
        correlation_1 = df_1.corr(method='pearson')['condConf_sizes']['predictions']
        print(f"Pearson correlation (condConf_sizes vs predictions): {correlation_1}")
        
        plt.figure(figsize=(10, 6))
        sns.regplot(x='condConf_sizes', y='predictions', data=df_1, scatter_kws={'s': 10}, line_kws={'color': 'red'})
        plt.title(f'Correlation between condConf Sizes and Predictions\nPearson correlation: {correlation_1:.2f}')
        plt.xlabel('condConf Sizes')
        plt.ylabel('Predictions')
        plt.figtext(0.5, -0.1, config_caption, ha="center", fontsize=10, wrap=True)
        plt.savefig(os.path.join(output_dir, 'condConf_sizes_vs_predictions.png'), bbox_inches="tight")
        plt.close()

    # 2. Scatter plot: condConf_sizes vs local 20*20 est. variances (skip if type isn't linear)
    if condconf_basis_type == "linear" and len(condConf_sizes) == len(results['variances']):
        variances = np.hstack([var.flatten() for var in results['variances']])
        df_2 = pd.DataFrame({
            'condConf_sizes': condConf_sizes,
            'variances': variances
        })
        correlation_2 = df_2.corr(method='pearson')['condConf_sizes']['variances']
        print(f"Pearson correlation (condConf_sizes vs variances): {correlation_2}")
        
        plt.figure(figsize=(10, 6))
        sns.regplot(x='condConf_sizes', y='variances', data=df_2, scatter_kws={'s': 10}, line_kws={'color': 'red'})
        plt.title(f'Correlation between condConf Sizes and Variances\nPearson correlation: {correlation_2:.2f}')
        plt.xlabel('condConf Sizes')
        plt.ylabel('Local 20*20 Estimated Variances')
        plt.figtext(0.5, -0.1, config_caption, ha="center", fontsize=10, wrap=True)
        plt.savefig(os.path.join(output_dir, 'condConf_sizes_vs_variances.png'), bbox_inches="tight")
        plt.close()

    # 3. Scatter plot: condConf_sizes vs rel_abs_error
    if len(condConf_sizes) == len(all_rel_abs_errors):
        df_3 = pd.DataFrame({
            'condConf_sizes': condConf_sizes,
            'rel_abs_error': all_rel_abs_errors
        })
        correlation_3 = df_3.corr(method='pearson')['condConf_sizes']['rel_abs_error']
        print(f"Pearson correlation (condConf_sizes vs rel_abs_error): {correlation_3}")
        
        plt.figure(figsize=(10, 6))
        sns.regplot(x='condConf_sizes', y='rel_abs_error', data=df_3, scatter_kws={'s': 10}, line_kws={'color': 'red'})
        plt.title(f'Correlation between condConf Sizes and Relative Absolute Errors\nPearson correlation: {correlation_3:.2f}')
        plt.xlabel('condConf Sizes')
        plt.ylabel('Relative Absolute Errors')
        plt.figtext(0.5, -0.1, config_caption, ha="center", fontsize=10, wrap=True)
        plt.savefig(os.path.join(output_dir, 'condConf_sizes_vs_rel_abs_error.png'), bbox_inches="tight")
        plt.close()


def crop_to_val_window_size(x):
  config = get_config()
  window_size = config["val_center_window_size"]
  print(f"Val img print window size: {window_size}")
  return get_center_window(x,window_size)

def create_colorbar(cmap, height, vmin, vmax, label, path):
    fig, ax = plt.subplots(figsize=(1, height / 100))  # Adjusted height to match image height
    fig.subplots_adjust(left=0.5, right=0.8)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cbar.ax.tick_params(labelsize=8)  # Customize tick label size
    cbar.ax.get_yaxis().labelpad = 15  # Adjust the padding

    # Set the tick marks for the range [-1, 1]
    if vmin == -1 and vmax == 1:
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['-1', '-0.5', '0', '0.5', '1'])

    fig.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def generate_config_caption():
    config = get_config()
    caption = f"Num calib img: {config['num_calib_img']}\n"
    caption += f"Calib img size: {config['center_window_size']}\n"
    caption += f"Val img size: {config['val_center_window_size']}\n"
    caption += f"CondConf basis: {config['condConf_basis_type']}"
    return caption


def plot_images_uq_modified(results,
                              weight_pred=0.4, weight_diff=0.6,
                              weight_gt=0.5, weight_abs_diff=0.5,
                              contrast_factor=1.5):
    # Assert that the pairs of weights add up to 1
    assert weight_pred + weight_diff == 1, "weight_pred and weight_diff must add up to 1"
    assert weight_gt + weight_abs_diff == 1, "weight_gt and weight_abs_diff must add up to 1"

    # Define colormaps
    coolwarm_cmap = cm.get_cmap('coolwarm', 50)
    wildfire_cmap = cmr.cm.wildfire  # Use the wildfire colormap from cmasher
    bwr_cmap = cm.get_cmap('bwr', 50)  # Use the bwr colormap from Matplotlib
    
    # generate directory
    config = get_config()
    base_output_path = get_img_save_fname(config)
    combined_all_path = os.path.join(base_output_path, "combined_all/")
    os.makedirs(combined_all_path, exist_ok=True)
    
    # set font for captions
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust the path to the font file if necessary
    font_size = 11  # Updated font size for all subtitles
    font = ImageFont.truetype(font_path, font_size)
    
    # generate val images' visualizations
    for i in range(len(results['predictions'])):
        foldername = os.path.join(base_output_path, f'{i}/')
        os.makedirs(foldername, exist_ok=True)
        # input
        input_image = normalize_01(crop_to_val_window_size(results['inputs'][i].squeeze()))
        # model prediction
        prediction = normalize_01(crop_to_val_window_size(results['predictions'][i].squeeze()))
        
        # condConf set sizes
        condConf_set_sizes = (results['condConf_upper_edge'][i] - results['condConf_lower_edge'][i]).squeeze()
        # vanilla CP set sizes
        CP_set_sizes = (results['CP_upper_edge'][i] - results['CP_lower_edge'][i]).squeeze()
        # CP UQ visualization
        mixed_CP_prediction_set_sizes = weight_diff * torch.tensor(coolwarm_cmap(normalize_01(CP_set_sizes.squeeze()))) + weight_pred * prediction.unsqueeze(2)
        # confConf UQ visualization
        mixed_condConf_prediction_set_sizes = weight_diff * torch.tensor(coolwarm_cmap(normalize_01(condConf_set_sizes.squeeze()))) + weight_pred * prediction.unsqueeze(2)
        # ground truth
        gt = normalize_01(crop_to_val_window_size(results['gt'][i].squeeze()))
        # errors
        diff = prediction - gt
        abs_diff = torch.abs(diff)
        # Compute the (relative) error normalized absolute difference
        relative_abs_diff = abs_diff / torch.max(torch.max(prediction), torch.max(gt))
        normalized_abs_diff = normalize_01(abs_diff)

        # Map the normalized signed difference to the wildfire colormap
        colored_diff = wildfire_cmap(normalize_01(diff))[:, :, :3]
        colored_diff_image = Image.fromarray((255 * colored_diff).astype('uint8')).convert('RGB')

        # Combine the colored signed difference with the prediction using weighted average
        combined_diff_pred = (weight_diff * colored_diff + weight_pred * prediction.unsqueeze(2).numpy())
        combined_diff_pred_image = Image.fromarray((255 * normalize_01(combined_diff_pred)).astype('uint8')).convert('RGB')

        # Create a colorbar for the combined_diff_pred image
        create_colorbar(coolwarm_cmap, colored_diff_image.height, 0, 1, 'Mixed Sizes Pred', os.path.join(foldername, "colorbar_mixed_sizes_pred.png"))

        # Save individual colorbars
        create_colorbar(wildfire_cmap, colored_diff_image.height, -1, 1, 'Difference Value', os.path.join(foldername, "colorbar_pred_gt.png"))
        create_colorbar(coolwarm_cmap, normalized_abs_diff.shape[0], 0, 1, 'Value', os.path.join(foldername, "colorbar_mixed.png"))

        # Create overlayed image
        overlayed_image = weight_gt * prediction + weight_abs_diff * normalized_abs_diff
        overlayed_image_normalized = normalize_01(overlayed_image)
        overlayed_image_colormap = coolwarm_cmap(overlayed_image_normalized.numpy())[:, :, :3]  # Ignore alpha channel
        # Additional images
        # bwr colormap applied to the ground truth image
        colored_gt = bwr_cmap(gt.numpy())[:, :, :3]
        colored_gt_image = Image.fromarray((255 * colored_gt).astype('uint8')).convert('RGB')
        colored_gt_image.save(os.path.join(foldername, "colored_gt.png"))
        create_colorbar(bwr_cmap, colored_gt_image.height, 0, 1, 'GT Value', os.path.join(foldername, "colorbar_colored_gt.png"))

        # Normalized absolute difference multiplied by the colormapped ground truth
        rel_diff_mult_colored_gt = normalize_01(relative_abs_diff.unsqueeze(2) * colored_gt).numpy()  # normalize_01(abs_diff).unsqueeze(2).numpy() * colored_gt
        rel_diff_mult_colored_gt_image = Image.fromarray((255 * rel_diff_mult_colored_gt).astype('uint8')).convert('RGB')
        rel_diff_mult_colored_gt_image.save(os.path.join(foldername, "rel_abs_diff * colored_gt.png"))

        # Save individual colorbars for additional images
        create_colorbar(bwr_cmap, rel_diff_mult_colored_gt_image.height, 0, 1, 'Rel Abs Diff * GT', os.path.join(foldername, "colorbar_rel_abs_diff_colored_gt.png"))

        # Generate and save colorbars for CP and condConf set sizes before accessing them
        create_colorbar(coolwarm_cmap, mixed_CP_prediction_set_sizes.shape[0], 0, 1, 'CP Sizes', os.path.join(foldername, "colorbar_CP_sizes.png"))
        create_colorbar(coolwarm_cmap, mixed_CP_prediction_set_sizes.shape[0], 0, 1, 'CondConf Sizes', os.path.join(foldername, "colorbar_condConf_sizes.png"))
       

        # If condConf_basis_type is not "linear", use the modified figure 10
        if config['condConf_basis_type'] != "linear":
            fig10_name = f"{weight_diff}*(pred-gt)+{weight_pred}*(pred)"
            fig10_image = (255 * combined_diff_pred).astype('uint8')
            grayscale_variances = None
        else:
            fig10_name = f"{weight_diff}*condConf_sizes+{weight_pred}*(pred)"
            fig10_image = (255 * normalize_01(mixed_condConf_prediction_set_sizes.cpu().numpy())).astype('uint8')
            variances = results['variances'][i].squeeze()  # Assuming variances is a tensor
            normalized_variances = normalize_01(variances)  # Normalize to [0, 1]
            grayscale_variances = (255 * normalized_variances.numpy()).astype('uint8')  # Convert to grayscale
        
        

        # Save image grid
        images = {

            # Fig 1-3 are for visualizing model predictions
            "gt": (255 * gt.numpy()).astype('uint8'), # Fig 1
            "input": (255 * input_image.numpy()).astype('uint8'), # Fig 2 
            "prediction": (255 * prediction.numpy()).astype('uint8'),  # Fig 3

            # Fig 4-6 are for visualizing how well different prediction sets captures model errors
            "abs_diff_normalized": (255 * normalized_abs_diff.numpy()).astype('uint8'), # Fig 4
            "condConf_set_sizes": (255 * normalize_01(condConf_set_sizes).numpy()).astype('uint8'), # Fig 5
            "CP_set_sizes": (255 * normalize_01(CP_set_sizes).numpy()).astype('uint8'), # Fig 6

            # Fig 7-9 are for additional model error visualizations
            "relative_abs_diff": (255 * relative_abs_diff.numpy()).astype('uint8'), # Fig 7
            "colored_gt": (255 * colored_gt).astype('uint8'),  # Fig 8
            "rel_abs_diff * colored_gt (Rina)": (255 * rel_diff_mult_colored_gt).astype('uint8'),  # Fig 9

            # Fig 10-12 are for additional UQ visualizations (order reversed)
            "grayscale_variances": grayscale_variances,  # Fig 10 (new)
            fig10_name: fig10_image,  # Adjusted Fig 10 depending on condConf_basis_type
            f"{weight_diff}*condConf_sizes+{weight_pred}*(pred)": (255 * normalize_01(mixed_condConf_prediction_set_sizes.cpu().numpy())).astype('uint8'),  # Fig 11
            f"{weight_diff}*CP_sizes+{weight_pred}*(pred)": (255 * normalize_01(mixed_CP_prediction_set_sizes.cpu().numpy())).astype('uint8')  # Fig 12
        }

        # Calculate frame width (same as label height)
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        label_height = max([temp_draw.textsize(name, font=font)[1] for name in images.keys()]) + 10

        # Adjust so that the grid is 4 rows * 3 columns with 100 extra pixels for the title
        img_width, img_height = images['gt'].shape[1], images['gt'].shape[0]
        total_width = 3 * (img_width + 2 * label_height + 50)  # make additional space for colorbars
        total_height = 4 * (img_height + 2 * label_height) + 100  # add 100 pixels for the main title
        grid_image = Image.new('RGB', (total_width, total_height), 'black')

        draw = ImageDraw.Draw(grid_image)

        # Positions of sub-images in each image grid, adjusted for the main title space
        positions = [
            (0, 100), (img_width + 2 * label_height + 50, 100), 
            (2 * (img_width + 2 * label_height + 50), 100),
            (0, img_height + 2 * label_height + 100),
            (img_width + 2 * label_height + 50, img_height + 2 * label_height + 100), 
            (2 * (img_width + 2 * label_height + 50), img_height + 2 * label_height + 100),
            (0, 2 * (img_height + 2 * label_height) + 100), 
            (img_width + 2 * label_height + 50, 2 * (img_height + 2 * label_height) + 100),
            (2 * (img_width + 2 * label_height + 50), 2 * (img_height + 2 * label_height) + 100),
            (0, 3 * (img_height + 2 * label_height) + 100), 
            (img_width + 2 * label_height + 50, 3 * (img_height + 2 * label_height) + 100),
            (2 * (img_width + 2 * label_height + 50), 3 * (img_height + 2 * label_height) + 100)
        ]
        
        names = [ 
            "gt", "input", "prediction",
            "abs_diff_normalized", "condConf_set_sizes", "CP_set_sizes",
            "relative_abs_diff", "colored_gt", "rel_abs_diff * colored_gt (Rina)",
            "grayscale_variances", f"{weight_diff}*condConf_sizes+{weight_pred}*(pred)", f"{weight_diff}*CP_sizes+{weight_pred}*(pred)"
        ]
        if config['condConf_basis_type'] != "linear":
          names[9] = fig10_name

        fig_numbers = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "Fig7", "Fig8", "Fig9", "Fig10", "Fig11", "Fig12"]

        for pos, name, fig_number in zip(positions, names, fig_numbers):
            sub_image = Image.fromarray(images[name]).convert('RGB')
            image_with_colorbar = sub_image
            

            # the following if statement adds appropriate colorbars to sub images as needed
            if name in ["colored_gt", fig10_name, f"{weight_diff}*CP_sizes+{weight_pred}*(pred)"]:
                if name == "colored_gt":
                    colorbar_path = os.path.join(foldername, "colorbar_colored_gt.png")
                elif name == fig10_name or (name == f"{weight_diff}*CP_sizes+{weight_pred}*(pred)"):
                    colorbar_path = os.path.join(foldername, "colorbar_mixed_sizes_pred.png")
                else:
                    colorbar_path = os.path.join(foldername, "colorbar_pred_gt.png")
                colorbar_image = Image.open(colorbar_path)
                colorbar_image = colorbar_image.resize((50, sub_image.height), Image.ANTIALIAS)  # Ensure the colorbar fits the image height
                image_with_colorbar = Image.new('RGB', (sub_image.width + colorbar_image.width, sub_image.height))
                image_with_colorbar.paste(sub_image, (0, 0))
                image_with_colorbar.paste(colorbar_image, (sub_image.width, 0))

            grid_image.paste(image_with_colorbar, (pos[0] + label_height, pos[1] + label_height))
            # Add label in the upper-left corner
            draw.text((pos[0] + 5, pos[1] + 5), name, font=font, fill="white")
            # Add figure number in the center of the bottom of the frame
            text_size = draw.textsize(fig_number, font=font)
            text_x = pos[0] + label_height + (img_width + label_height) // 2 - text_size[0] // 2
            text_y = pos[1] + label_height + img_height + 5
            draw.text((text_x, text_y), fig_number, font=font, fill="white")

        # Add the main title at the top of the image grid, with descriptions of each parameter in separate lines
        config_caption = generate_config_caption()
        title_font_size = 20
        title_font = ImageFont.truetype(font_path, title_font_size)
        title_size = draw.textsize(config_caption, font=title_font)
        title_x = (grid_image.width - title_size[0]) // 2
        draw.text((title_x, 10), config_caption, font=title_font, fill="white")

        # Save the grid image with colorbars
        grid_image_path = foldername + "combined_with_colorbar.png"
        grid_image.save(grid_image_path)

        # Copy the grid image with colorbars to the combined_all folder
        shutil.copy(grid_image_path, os.path.join(combined_all_path, f"{i}_combined_with_colorbar.png"))

def plot_images_uq(results):
  uq_cmap = cm.get_cmap('coolwarm',50)
  os.makedirs('outputs/images/',exist_ok=True)
  for i in range(len(results['predictions'])):   
    foldername = f'outputs/images/{i}/'
    os.makedirs(foldername,exist_ok=True)
    input_image = normalize_01(results['inputs'][i].squeeze())
    prediction = normalize_01(results['predictions'][i].squeeze())
    condConf_set_sizes = (results['upper_edge'][i] - results['lower_edge'][i]).squeeze()
    mixed_output = 0.3*torch.tensor(uq_cmap(normalize_01(set_sizes.squeeze()))) + 0.7*prediction.unsqueeze(2)
    im = Image.fromarray((255*input_image.numpy()).astype('uint8')).convert('RGB')
    im.save(foldername + "input.png")
    im = Image.fromarray((255*prediction.numpy()).astype('uint8')).convert('RGB')
    im.save(foldername + "prediction.png")
    im = Image.fromarray((255*normalize_01(set_sizes).numpy()).astype('uint8')).convert('RGB')
    im.save(foldername + "set_sizes.png")
    im = Image.fromarray((255*normalize_01(results['gt'][i].squeeze()).numpy()).astype('uint8')).convert('RGB')
    im.save(foldername + "gt.png")
    im = Image.fromarray((255*mixed_output.numpy()).astype('uint8')).convert('RGB')
    im.save(foldername + "mixed_output.png")

def plot_spatial_miscoverage(methodnames, results_list):
  plt.figure(figsize=(5,5))
  #sns.set(font_scale=1.2) # 1col scaling
  sns.set(font_scale=2) # 2col scaling
  sns.set_style("white")
  sns.set_palette('pastel')
  uq_cmap = cm.get_cmap('coolwarm',50)
  foldername = 'outputs/spatial_miscoverage/'
  os.makedirs(foldername,exist_ok=True)
  for i in range(len(results_list)): 
    spatial_miscoverage = results_list[i]['spatial_miscoverage']
    im = Image.fromarray((255*uq_cmap(spatial_miscoverage)).astype('uint8')).convert('RGB')
    im.save(foldername + f"fastMRI_spatial_miscoverage_{methodnames[i]}.png")

def generate_plots():
  methodnames = ['Quantile Regression']

  config = get_config()
  results_filenames = [get_img_generating_fname(config)] 
  #results_filenames = ['/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_test/outputs/raw/results_fastmri_quantiles_16_0.001_standard_(min-max)_50_100_const.pkl']
  print(results_filenames)
  #results_filenames = ['/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_test/outputs/raw/results_fastmri_quantiles_16_0.001_standard_(min-max)_50_50.pkl']
  loss_tables_filenames = ['/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_test/outputs/raw/loss_table_fastmri_quantiles_16_0.001_standard_min-max.pth']
  
  #methodnames = ['Residual Magnitude','Gaussian','Softmax','Quantile Regression']
  #results_filenames = ['outputs/raw/results_fastmri_residual_magnitude_78_0.0001_standard_standard.pkl','outputs/raw/results_fastmri_gaussian_78_0.001_standard_standard.pkl','outputs/raw/results_fastmri_softmax_256_0.001_standard_min-max.pkl','outputs/raw/results_fastmri_quantiles_78_0.0001_standard_standard.pkl']
  #loss_tables_filenames = ['outputs/raw/loss_table_fastmri_residual_magnitude_78_0.0001_standard_standard.pth','outputs/raw/loss_table_fastmri_gaussian_78_0.001_standard_standard.pth','outputs/raw/loss_table_fastmri_softmax_256_0.001_standard_min-max.pth','outputs/raw/loss_table_fastmri_quantiles_78_0.0001_standard_standard.pth']
  # The max and std of the dataset are needed to rescale the MSE and set size properly for _standard
  dataset_std = 7.01926983310841e-05
  dataset_max = 0.0026554432697594166
  # Load results
  results_list = []
  for filename in results_filenames:
    with open(filename, 'rb') as handle:
      result = CPU_Unpickler(handle).load()
      if 'standard_standard' in filename:
        result['mse'] = (result['mse'] * dataset_std) / dataset_max
        result['sizes'] = (result['sizes'] * dataset_std) / dataset_max
      results_list = results_list + [result,]
  loss_tables_list = []
  for filename in loss_tables_filenames:
    loss_tables_list = loss_tables_list + [torch.load(filename),]
  alpha = 0.1
  delta = 0.1
  n = loss_tables_list[0].shape[0]//2
  # Plot spatial miscoverage
  ### plot_spatial_miscoverage(methodnames, results_list)
  # Plot mse
  ### plot_mse(methodnames,results_list)
  # Plot risks
  ### plot_risks(methodnames,loss_tables_list,n,alpha,delta)
  # Plot spearman correlations
  ### plot_spearman(methodnames,results_list)
  # Plot size-stratified risks 
  ### plot_ssr(methodnames,results_list,alpha)
  # Plot size distribution
  #### plot_size_violins(methodnames,results_list)
  # Plot the MRI images (only quantile regression)
  ###plot_images_uq(results_list[-1])
  plot_images_uq_modified(results_list[-1])
  plot_correlation_from_results(results_list[-1])

if __name__ == "__main__":
  generate_plots()
