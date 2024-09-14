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
  plt.savefig('/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/fastmri-mse.pdf',bbox_inches="tight")

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
  plt.savefig('/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/fastmri-spearman.pdf',bbox_inches="tight")

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
  plt.savefig('/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/fastmri-sizes.pdf',bbox_inches="tight")

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
  plt.savefig('/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/fastmri-size-stratified-risk.pdf',bbox_inches="tight")

def plot_risks(methodnames,loss_table_list,n,alpha,delta,num_trials=100): 
  fname = '/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/raw/risks.pth'
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
  plt.savefig('/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/fastmri-risks.pdf',bbox_inches="tight")



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

def plot_images_uq(results, weight_pred=0.4, weight_diff=0.6, weight_gt=0.5, weight_abs_diff=0.5, contrast_factor=1.5):
    # Assert that the pairs of weights add up to 1
    assert weight_pred + weight_diff == 1, "weight_pred and weight_diff must add up to 1"
    assert weight_gt + weight_abs_diff == 1, "weight_gt and weight_abs_diff must add up to 1"

    # Define colormaps
    coolwarm_cmap = cm.get_cmap('coolwarm', 50)
    wildfire_cmap = cmr.cm.wildfire  # Use the wildfire colormap from cmasher
    bwr_cmap = cm.get_cmap('bwr', 50)  # Use the bwr colormap from Matplotlib
    
    base_output_path = '/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/images/'
    combined_all_path = os.path.join(base_output_path, 'combined_all/')
    os.makedirs(combined_all_path, exist_ok=True)
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust the path to the font file if necessary
    font_size = 14
    font = ImageFont.truetype(font_path, font_size)
    
    for i in range(len(results['predictions'])):
        foldername = os.path.join(base_output_path, f'{i}/')
        os.makedirs(foldername, exist_ok=True)
        input_image = normalize_01(results['inputs'][i].squeeze())
        prediction = normalize_01(results['predictions'][i].squeeze())
        set_sizes = (results['upper_edge'][i] - results['lower_edge'][i]).squeeze()
        mixed_prediction_set_sizes = weight_pred * torch.tensor(coolwarm_cmap(normalize_01(set_sizes.squeeze()))) + weight_diff * prediction.unsqueeze(2)

        gt = normalize_01(results['gt'][i].squeeze())
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


        # Save image grid
        images = {
            "gt": (255 * gt.numpy()).astype('uint8'),
            "input": (255 * input_image.numpy()).astype('uint8'),
            "prediction": (255 * prediction.numpy()).astype('uint8'),
            "set_sizes": (255 * normalize_01(set_sizes).numpy()).astype('uint8'),
            "relative_abs_diff": (255 * relative_abs_diff.numpy()).astype('uint8'),
            "|gt-pred|(normalized)": (255 * normalized_abs_diff.numpy()).astype('uint8'),
            "(pred-gt)colormap": (255 * colored_diff).astype('uint8'),
            f"{weight_diff}*(pred-gt)+{weight_pred}*(pred)": (255 * combined_diff_pred).astype('uint8'),
            "colored_gt": (255 * colored_gt).astype('uint8')
        }

        
        # Normalized absolute difference multiplied by the colormapped ground truth
        abs_diff_colored_gt = normalize_01(abs_diff).unsqueeze(2).numpy() * colored_gt
        abs_diff_colored_gt_image = Image.fromarray((255 * abs_diff_colored_gt).astype('uint8')).convert('RGB')
        abs_diff_colored_gt_image.save(os.path.join(foldername, "|gt-pred| * colored_gt.png"))

        # Save individual colorbars for additional images
        create_colorbar(bwr_cmap, colored_gt_image.height, 0, 1, 'GT Value', os.path.join(foldername, "colorbar_colored_gt.png"))

        images["colored_gt"] = (255 * colored_gt).astype('uint8')
        images["|gt-pred| * colored_gt"] = (255 * abs_diff_colored_gt).astype('uint8')

        # Calculate frame width (same as label height)
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        label_height = max([temp_draw.textsize(name, font=font)[1] for name in images.keys()]) + 10

        # Create the 2x5 grid image with additional space for frames
        img_width, img_height = images['gt'].shape[1], images['gt'].shape[0]
        total_width = 5 * (img_width + 2 * label_height + 50)  # Space for colorbars
        total_height = 2 * (img_height + 2 * label_height)
        grid_image = Image.new('RGB', (total_width, total_height), 'black')

        draw = ImageDraw.Draw(grid_image)

        positions = [
            (0, 0), (img_width + 2 * label_height + 50, 0), (2 * (img_width + 2 * label_height + 50), 0), (3 * (img_width + 2 * label_height + 50), 0), (4 * (img_width + 2 * label_height + 50), 0),
            (0, img_height + 2 * label_height), (img_width + 2 * label_height + 50, img_height + 2 * label_height), 
            (2 * (img_width + 2 * label_height + 50), img_height + 2 * label_height), (3 * (img_width + 2 * label_height + 50), img_height + 2 * label_height), (4 * (img_width + 2 * label_height + 50), img_height + 2 * label_height)
        ]
        names = ["gt", "input", "prediction", "set_sizes", "relative_abs_diff", "|gt-pred|(normalized)", f"{weight_diff}*(pred-gt)+{weight_pred}*(pred)", "(pred-gt)colormap", "colored_gt", "|gt-pred| * colored_gt"]

        fig_numbers = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "Fig7", "Fig8", "Fig9", "Fig10"]

        for pos, name, fig_number in zip(positions, names, fig_numbers):
            sub_image = Image.fromarray(images[name]).convert('RGB')
            image_with_colorbar = sub_image

            if name in ["(pred-gt)colormap", "mixed_prediction_set_sizes", "colored_gt"]:
                if name == "(pred-gt)colormap":
                    colorbar_path = os.path.join(foldername, "colorbar_pred_gt.png")
                elif name == "colored_gt":
                    colorbar_path = os.path.join(foldername, "colorbar_colored_gt.png")
                else:
                    colorbar_path = os.path.join(foldername, "colorbar_mixed.png")
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

        # Save the grid image with colorbars
        grid_image_path = foldername + "combined_with_colorbar.png"
        grid_image.save(grid_image_path)

        # Copy the grid image with colorbars to the combined_all folder
        shutil.copy(grid_image_path, os.path.join(combined_all_path, f"{i}_combined_with_colorbar.png"))

def plot_spatial_miscoverage(methodnames, results_list):
  plt.figure(figsize=(5,5))
  #sns.set(font_scale=1.2) # 1col scaling
  sns.set(font_scale=2) # 2col scaling
  sns.set_style("white")
  sns.set_palette('pastel')
  uq_cmap = cm.get_cmap('coolwarm',50)
  foldername = '/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/spatial_miscoverage/'
  os.makedirs(foldername,exist_ok=True)
  for i in range(len(results_list)): 
    spatial_miscoverage = results_list[i]['spatial_miscoverage']
    im = Image.fromarray((255*uq_cmap(spatial_miscoverage)).astype('uint8')).convert('RGB')
    im.save(foldername + f"fastMRI_spatial_miscoverage_{methodnames[i]}.png")

def generate_plots():
  methodnames = ['Quantile Regression']
  #['Residual Magnitude', 'Gaussian', 'Softmax', 'Quantile Regression']
  results_filenames = ['/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/raw/results_fastmri_quantiles_16_0.001_standard_min-max.pkl']
  #['outputs/raw/results_fastmri_quantiles_16_0.001_standard_min-max.pkl']#['outputs/raw/results_fastmri_residual_magnitude_78_0.0001_standard_standard.pkl', 'outputs/raw/results_fastmri_gaussian_78_0.0001_standard_standard.pkl', 'outputs/raw/results_fastmri_softmax_64_0.001_standard_min-max.pkl', 'outputs/raw/results_fastmri_quantiles_16_0.001_standard_standard.pkl']
  loss_tables_filenames =  ['/project2/rina/lekunbillwang/im2im-uq/experiments/fastmri_train/outputs/raw/loss_table_fastmri_quantiles_16_0.001_standard_min-max.pth']
   #['outputs/raw/loss_table_fastmri_residual_magnitude_78_0.0001_standard_standard.pth', 'outputs/raw/loss_table_fastmri_gaussian_78_0.0001_standard_standard.pth', 'outputs/raw/loss_table_fastmri_softmax_64_0.001_standard_min-max.pth', 'outputs/raw/loss_table_fastmri_quantiles_16_0.001_standard_standard.pth']
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
  plot_spatial_miscoverage(methodnames, results_list)
  # Plot mse
  plot_mse(methodnames,results_list)
  # Plot risks
  plot_risks(methodnames,loss_tables_list,n,alpha,delta)
  # Plot spearman correlations
  plot_spearman(methodnames,results_list)
  # Plot size-stratified risks 
  plot_ssr(methodnames,results_list,alpha)
  # Plot size distribution
  plot_size_violins(methodnames,results_list)
  # Plot the MRI images (only quantile regression)
  plot_images_uq(results_list[-1])

if __name__ == "__main__":
  generate_plots()
