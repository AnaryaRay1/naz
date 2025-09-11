#!/bin/bash
#SBATCH -A b1094               # Allocation
#SBATCH -p ciera-gpu     # Queue
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00             # Walltime/duration of the job
#SBATCH --mem=100G          
#SBATCH --error=./nf_maf_time_10q_log.err
#SBATCH --output=./nf_maf_time_10q_log.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizabethteng@u.northwestern.edu

source ~/miniconda/etc/profile.d/conda.sh
# module load mamba/24.3.0
source activate naz

python train_maf_mle_q.py --fthin=1 --index=0 --popsynth-file=/projects/b1119/eteng/popsynth/pops/default_cat_minus_default_pop_time_10k_q.h5 --epistemic-only=True --nhidden=512 --nlayer=5 --nflow=16 --dir='MAF5_time_10k_q_log'
