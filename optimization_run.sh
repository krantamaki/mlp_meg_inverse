#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -n 1

module load neuroimaging

srun python main.py
