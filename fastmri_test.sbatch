#!/bin/bash
#SBATCH --job-name=singlecoil_val_replicate
#SBATCH --output=job-info.out
#SBATCH --error=job-info.err
#SBATCH --account=pi-rina
#SBATCH --partition=gpu2
#SBATCH --time=00:10:30
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=lekunbillwang@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

# LOAD MODULES
module load python

# ACTIVATE ENVIRONMENT
source activate im2im-uq

# COMPUTE!
python core/scripts/router.py