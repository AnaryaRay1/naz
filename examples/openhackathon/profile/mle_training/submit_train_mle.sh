#!/usr/bin/bash

#SBATCH --account=bccu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=100G 
#SBATCH --gpu-bind=single:1
#SBATCH --time=00:20:00
#SBATCH --job-name=train_noprofile
#SBATCH --output=__logs__/train.out
#SBATCH --error=__logs__/train.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.teng@northwestern.edu

./train_mle_q.sh
