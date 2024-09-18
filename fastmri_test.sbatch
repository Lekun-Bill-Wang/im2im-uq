#!/bin/bash
#SBATCH --job-name=singlecoil_train_replicate
#SBATCH --output=singlecoil_train_replicate.out
#SBATCH --error=singlecoil_train_replicate.err
#SBATCH --account=pi-rina
#SBATCH --partition=gpu2
#SBATCH --time=23:59:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=28
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=lekunbillwang@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

# LOAD MODULES
module load python

# ACTIVATE ENVIRONMENT
source activate im2im-uq

# COMPUTE!
python core/scripts/router.py
