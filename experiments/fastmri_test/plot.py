import os, sys, io, pathlib
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from core.calibration.calibrate_model import evaluate_from_loss_table
from core.scripts.eval import transform_output 
from tqdm import tqdm
from PIL import Image
import pdb

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
  sns.set(font_scale=1.35)
  sns.set_style("white")
  sns.set_palette('pastel')
  # Crop sizes to 99%
  for results in results_list:
    results['sizes'] = torch.clamp(results['sizes'], min=0, max=2)
  df = pd.DataFrame({'Interval Size' : torch.cat([results['sizes'] for results in results_list]).tolist(), 'Method': [method.replace(' ','\n') for method in methodnames for i in range(results_list[0]['sizes'].shape[0])]})
  g = sns.violinplot(data=df, x='Method', y='Interval Size', cut=0)
  sns.despine(top=True, right=True)
  plt.yticks([0,1,2])
  plt.xlabel('')
  plt.gca().set_yticklabels(['0%','50%','100%'])
  plt.tight_layout()
  plt.savefig('outputs/fastmri-sizes.pdf',bbox_inches="tight")

def plot_ssr(methodnames,results_list):
  plt.figure(figsize=(4,4))
  sns.set(font_scale=1.35)
  sns.set_style("white")
  sns.set_palette(sns.light_palette("salmon"))
  df = pd.DataFrame({'Difficulty': len(results_list)*['Easy', 'Easy-Medium', 'Medium-Hard', 'Hard'], 'Risk' : torch.cat([results['size-stratified risk'] for results in results_list]).tolist(), 'Method': [method.replace(' ','\n') for method in methodnames for i in range(results_list[0]['size-stratified risk'].shape[0])]})
  g = sns.catplot(data=df, kind='bar', x='Method', y='Risk', hue='Difficulty',legend=False)
  sns.despine(top=True, right=True)
  plt.legend(loc='upper right')
  plt.xlabel('')
  plt.locator_params(axis="y", nbins=5)
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
  sns.set(font_scale=1.35)
  sns.set_style("white")
  sns.set_palette('pastel')
  df = pd.DataFrame({'Method' : [method.replace(' ','\n') for method in methodnames for i in range(num_trials)], 'Risk' : torch.cat(risks_list,dim=0).tolist()})
  g = sns.violinplot(data=df, x='Method', y='Risk')
  plt.gca().axhline(y=alpha, color='#888888', linewidth=2, linestyle='dashed')
  sns.despine(top=True, right=True)
  plt.ylim([0.07,None])
  plt.xlabel('')
  plt.locator_params(axis="y", nbins=5)
  plt.tight_layout()
  plt.savefig('outputs/fastmri-risks.pdf',bbox_inches="tight")

def plot_images_uq(results):
  uq_cmap = cm.get_cmap('coolwarm',50)
  os.makedirs('outputs/images/',exist_ok=True)
  for i in range(len(results['predictions'])):   
    foldername = f'outputs/images/{i}/'
    os.makedirs(foldername,exist_ok=True)
    input_image = normalize_01(results['inputs'][i].squeeze())
    prediction = normalize_01(results['predictions'][i].squeeze())
    set_sizes = (results['upper_edge'][i] - results['lower_edge'][i]).squeeze()
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

def generate_plots():
  methodnames = ['Gaussian','Residual Magnitude','Quantile Regression']
  results_filenames = ['outputs/raw/results_fastmri_gaussian_78_0.0001_standard_standard.pkl','outputs/raw/results_fastmri_residual_magnitude_78_0.0001_standard_standard.pkl','outputs/raw/results_fastmri_quantiles_78_0.0001_standard_standard.pkl']
  loss_tables_filenames = ['outputs/raw/loss_table_fastmri_gaussian_78_0.0001_standard_standard.pth','outputs/raw/loss_table_fastmri_residual_magnitude_78_0.0001_standard_standard.pth','outputs/raw/loss_table_fastmri_quantiles_78_0.0001_standard_standard.pth']
  # Load results
  results_list = []
  for filename in results_filenames:
    with open(filename, 'rb') as handle:
      results_list = results_list + [CPU_Unpickler(handle).load(),]
  loss_tables_list = []
  for filename in loss_tables_filenames:
    loss_tables_list = loss_tables_list + [torch.load(filename),]
  # Plot risks
  alpha = 0.1
  delta = 0.1
  n = loss_tables_list[0].shape[0]//2
  plot_risks(methodnames,loss_tables_list,n,alpha,delta)
  # Plot spearman correlations
  plot_spearman(methodnames,results_list)
  # Plot size-stratified risks 
  plot_ssr(methodnames,results_list)
  # Plot size distribution
  plot_size_violins(methodnames,results_list)
  # Plot the MRI images (only quantile regression)
  plot_images_uq(results_list[-1])

if __name__ == "__main__":
  generate_plots()
