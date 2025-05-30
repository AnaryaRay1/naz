#!/usr/bin/bash

#SBATCH --account=p32465
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem-per-gpu=100G 
#SBATCH --gpu-bind=single:1
#SBATCH --time=16:00:00
#SBATCH --job-name=train
#SBATCH --output=__logs__/hmc_h100.out
#SBATCH --error=__logs__/hmc_h100.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anarya.ray@northwestern.edu


conda activate naz-prod
nvidia-smi
python hmc_maf_exact.py --fthin=100 --num-warmup=100 --num-samples=1800 --mle-flow=inference_mle_01_1_prod_2d_150_3_16_mcchi.pkl --sigma=0.35 --chckpt=True
