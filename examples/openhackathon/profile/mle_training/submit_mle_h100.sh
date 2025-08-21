#!/usr/bin/bash

#SBATCH --account=bccu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem-per-gpu=100G 
#SBATCH --gpu-bind=closest
#SBATCH --time=00:20:00
#SBATCH --job-name=profile
#SBATCH --output=__logs__/maf_50_10k_float16.out
#SBATCH --error=__logs__/maf_50_10k_float16.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.teng@northwestern.edu

./run_mle.sh
