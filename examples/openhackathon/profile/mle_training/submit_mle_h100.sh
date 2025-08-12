#!/usr/bin/bash

#SBATCH --account=bccu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=100G 
#SBATCH --gpu-bind=single:1
#SBATCH --time=00:20:00
#SBATCH --job-name=train
#SBATCH --output=__logs__/hmc_500_025_2p_profile.out
#SBATCH --error=__logs__/hmc_500_025_2p_profile.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.teng@northwestern.edu

./run_mle.sh
