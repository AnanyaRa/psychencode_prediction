#!/bin/bash

#SBATCH --partition=pi_gerstein
#SBATCH --job-name=L_22_job
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=3:00:00
#SBATCH --mail-type ALL

module purge
module load miniconda
conda activate thesis_env

time python3 rf_celltype_covariates.py

