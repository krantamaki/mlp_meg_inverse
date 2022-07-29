#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH --time=07-00
#SBATCH --mail-type=all
#SBATCH --mail-user=kasper.rantamaki@aalto.fi
#SBATCH --mem-per-cpu=4G
#SBATCH -n 1

module load neuroimaging

srun python main.py
