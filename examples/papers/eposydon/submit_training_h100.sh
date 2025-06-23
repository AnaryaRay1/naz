#!/usr/bin/bash

#SBATCH --account=p32465
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=100G 
#SBATCH --gpu-bind=single:1
#SBATCH --time=16:00:00
#SBATCH --job-name=train
#SBATCH --output=logs_new/outlog4_mle.out
#SBATCH --error=logs_new/outlog4_mle.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anarya.ray@northwestern.edu

source ~/miniconda/etc/profile.d/conda.sh
conda activate naz-prod2
nvidia-smi
python train_maf_mle.py --fthin=1 --index=0 --popsynth-file=/projects/b1094/eteng/default_cat.h5 --epistemic-only=True --nhidden=512 --nlayer=5 --nflow=16
