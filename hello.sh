#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=100M
#SBATCH --output=hello.out

srun echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."